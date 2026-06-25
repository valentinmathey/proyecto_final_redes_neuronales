from __future__ import annotations
from contextlib import contextmanager
import copy
from datetime import datetime
from pathlib import Path
from itertools import islice
import torch
from tqdm.auto import tqdm

from .detection_metrics import evaluate_map, extract_main_map_metrics

# Fuerza modo eval solo durante el bloque y luego restaura el estado previo.
@contextmanager
def eval_mode(model):
    was_training = model.training
    model.eval()
    try:
        yield model
    finally:
        if was_training:
            model.train()

# Fuerza modo train solo durante el bloque y luego restaura el estado previo.
@contextmanager
def train_mode(model):
    was_training = model.training
    model.train()
    try:
        yield model
    finally:
        if not was_training:
            model.eval()

# Mueve imágenes y tensores de target al dispositivo elegido.
def move_batch_to_device(images, targets, device):
    images = [image.to(device) for image in images]
    targets = [
        {k: v.to(device) if torch.is_tensor(v) else v for k, v in t.items()}
        for t in targets
    ]
    return images, targets

# Imprime logs solo cuando el experimento está en modo verbose.
def _log(experiment_name, message, verbose):
    if verbose:
        print(f"[{experiment_name}] {message}")


@contextmanager
def _timed():
    # Mide hora de inicio, fin y duración del bloque encapsulado.
    started_at = datetime.now()
    result = {"start": started_at}
    yield result
    finished_at = datetime.now()
    result["end"] = finished_at
    result["duration_seconds"] = (finished_at - started_at).total_seconds()
# Ejecuta una época completa de entrenamiento y devuelve la loss promedio.
def train_one_epoch(model, dataloader, optimizer, device, epoch_index, max_batches=None):
    model.train()
    running_loss = 0.0
    running_steps = 0

    total_batches = len(dataloader)
    if max_batches is not None:
        total_batches = min(total_batches, max_batches)

    progress_bar = tqdm(
        dataloader,
        total=total_batches,
        desc=f"Train Epoch {epoch_index}",
        leave=False,
    )

    for images, targets in islice(progress_bar, max_batches):
        # Prepara el batch en GPU/CPU antes del forward.
        images, targets = move_batch_to_device(images, targets, device)

        # Los detectores de torchvision devuelven un dict con varias losses parciales.
        loss_dict = model(images, targets)
        total_loss = sum(loss for loss in loss_dict.values())

        # Backprop clásico sobre la loss agregada del batch.
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        running_loss += float(total_loss.item())
        running_steps += 1
        progress_bar.set_postfix(
            batch_loss=f"{float(total_loss.item()):.4f}",
            avg_loss=f"{running_loss / running_steps:.4f}",
        )

    progress_bar.close()

    average_loss = running_loss / max(running_steps, 1)
    return {
        "epoch": epoch_index,
        "train_loss": average_loss,
        "train_steps": running_steps,
    }

# Calcula loss de validación sin actualizar pesos ni construir grafo.
def evaluate_detection_loss(model, dataloader, device, max_batches=None):
    total_batches = min(len(dataloader), max_batches or len(dataloader))
    running_loss = 0.0
    running_steps = 0

    # torchvision solo devuelve losses con targets cuando el modelo está en train().
    with train_mode(model), torch.no_grad():
        progress_bar = tqdm(dataloader, total=total_batches, desc="Validation Loss", leave=False)
        for images, targets in islice(progress_bar, max_batches):
            images, targets = move_batch_to_device(images, targets, device)
            loss_dict = model(images, targets)
            total_loss = sum(loss_dict.values())

            running_loss += float(total_loss.item())
            running_steps += 1
            progress_bar.set_postfix(
                batch_loss=f"{float(total_loss.item()):.4f}",
                avg_loss=f"{running_loss / running_steps:.4f}",
            )

        progress_bar.close()

    return {
        "val_loss": running_loss / max(running_steps, 1),
        "val_steps": running_steps,
    }

# Empaqueta el estado suficiente para reanudar o evaluar una corrida.
def _checkpoint_payload(model, optimizer, scheduler, history, epoch, experiment_name, config):
    payload = {
        "experiment_name": experiment_name,
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "history": history,
        "config": config,
    }

    if scheduler is not None:
        payload["scheduler_state_dict"] = scheduler.state_dict()

    return payload

# Orquesta entrenamiento, validación, mAP y persistencia del mejor checkpoint.
def run_detection_experiment(
    model,
    train_loader,
    val_loader,
    optimizer,
    device,
    num_epochs,
    experiment_name,
    config,
    output_dir=None,
    max_train_batches=None,
    max_val_batches=None,
    class_metrics=True,
    verbose=True,
):
    history = []
    best_metric = float("-inf")
    best_epoch = None
    best_checkpoint_path = None
    best_payload = None

    output_path = Path(output_dir) if output_dir is not None else None
    if output_path is not None:
        # Crea la carpeta de artefactos si el experimento debe persistir checkpoints.
        output_path.mkdir(parents=True, exist_ok=True)

    model.to(device)

    with _timed() as training_time:
        _log(experiment_name, f"Starting training for {num_epochs} epoch(s) at {training_time['start'].isoformat(timespec='seconds')}", verbose)

        for epoch in range(1, num_epochs + 1):
            with _timed() as epoch_time:
                _log(experiment_name, f"Epoch {epoch}/{num_epochs} started at {epoch_time['start'].isoformat(timespec='seconds')}", verbose)

                train_metrics = train_one_epoch(
                    model=model,
                    dataloader=train_loader,
                    optimizer=optimizer,
                    device=device,
                    epoch_index=epoch,
                    max_batches=max_train_batches,
                )
                val_loss_metrics = evaluate_detection_loss(
                    model=model,
                    dataloader=val_loader,
                    device=device,
                    max_batches=max_val_batches,
                )
                map_metrics = evaluate_map(
                    model=model,
                    dataloader=val_loader,
                    device=device,
                    class_metrics=class_metrics,
                    max_batches=max_val_batches,
                )

            epoch_metrics = {
                "epoch": epoch,
                "epoch_start_time": epoch_time["start"].isoformat(timespec="seconds"),
                "epoch_end_time": epoch_time["end"].isoformat(timespec="seconds"),
                "epoch_duration_seconds": epoch_time["duration_seconds"],
                "train_loss": train_metrics["train_loss"],
                "val_loss": val_loss_metrics["val_loss"],
                "lr": float(optimizer.param_groups[0]["lr"]),
                **extract_main_map_metrics(map_metrics),
            }
            history.append(epoch_metrics)

            _log(experiment_name,
                f"Epoch {epoch}/{num_epochs} - "
                f"duration_s={epoch_metrics['epoch_duration_seconds']:.2f} - "
                f"train_loss={epoch_metrics['train_loss']:.4f} - "
                f"val_loss={epoch_metrics['val_loss']:.4f} - "
                f"map={epoch_metrics.get('map', 0.0):.4f} - "
                f"map_50={epoch_metrics.get('map_50', 0.0):.4f} - "
                f"lr={epoch_metrics['lr']:.6f}",
                verbose,
            )

            current_metric = epoch_metrics.get("map")
            if current_metric is not None and current_metric > best_metric:
                # Cuando mejora el mAP, se guarda una instantánea completa del experimento.
                best_metric = current_metric
                best_epoch = epoch
                best_payload = _checkpoint_payload(
                    model=model,
                    optimizer=optimizer,
                    scheduler=None,
                    history=copy.deepcopy(history),
                    epoch=epoch,
                    experiment_name=experiment_name,
                    config=config,
                )
                if output_path is not None:
                    best_checkpoint_path = output_path / f"{experiment_name}_best.pth"
                    torch.save(best_payload, best_checkpoint_path)
                _log(experiment_name, f"New best checkpoint at epoch {epoch} with map={best_metric:.4f}", verbose)

    _log(experiment_name,
        f"Training finished - "
        f"duration_s={training_time['duration_seconds']:.2f} - "
        f"best_epoch={best_epoch} - best_map={best_metric:.4f}",
        verbose,
    )

    return {
        "history": history,
        "best_metric": best_metric,
        "best_epoch": best_epoch,
        "best_checkpoint_path": str(best_checkpoint_path) if best_checkpoint_path else None,
        "best_payload": best_payload,
        "training_start_time": training_time["start"].isoformat(timespec="seconds"),
        "training_end_time": training_time["end"].isoformat(timespec="seconds"),
        "training_duration_seconds": training_time["duration_seconds"],
    }

# Restaura el state_dict del modelo a partir de un checkpoint confiable del proyecto.
def load_checkpoint(model, checkpoint_path, device="cpu"):
    # Estos checkpoints los genera el propio proyecto con torch.save(...)
    # y contienen mas que solo tensores (config, history, optimizer state).
    # En PyTorch 2.6 torch.load usa weights_only=True por defecto, lo que
    # rompe este formato. Forzamos weights_only=False para restaurarlos.
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    return checkpoint
