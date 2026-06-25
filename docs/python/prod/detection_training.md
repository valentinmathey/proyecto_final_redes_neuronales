# `prod/detection_training.py`

## Proposito del archivo

Este modulo concentra la logica reusable de entrenamiento y validacion para deteccion.

Incluye:

- utilidades de modo evaluacion
- utilidades de modo entrenamiento temporal
- movimiento de batches a dispositivo
- medicion de tiempos
- entrenamiento por epoca
- evaluacion de loss
- experimento completo con checkpoints
- carga de checkpoints

## Context managers y helpers

### `eval_mode(model)`

Context manager que pone al modelo en modo evaluacion durante un bloque y luego restaura el modo anterior.

Se usa para evaluar sin perder el estado previo de training.

### `move_batch_to_device(images, targets, device)`

Mueve a CPU o GPU:

- cada tensor de `images`
- cada tensor dentro de los diccionarios `targets`

Devuelve el batch transformado.

### `_log(experiment_name, message, verbose)`

Helper interno de logging condicional.

Solo imprime si `verbose=True`.

### `_timed()`

Context manager interno para medir duracion.

Expone un diccionario con:

- `start`
- `end`
- `duration_seconds`

Se usa para medir tanto el entrenamiento completo como cada epoca.

## Entrenamiento y validacion

### `train_one_epoch(model, dataloader, optimizer, device, epoch_index, max_batches=None)`

Ejecuta una epoca de entrenamiento.

Flujo principal:

1. pone el modelo en modo train
2. recorre el `dataloader`
3. mueve el batch al dispositivo
4. hace forward con `targets`
5. suma las losses devueltas por el modelo
6. hace backward y `optimizer.step()`
7. acumula promedio de loss

Devuelve un diccionario con:

- `epoch`
- `train_loss`
- `train_steps`

`max_batches` permite limitar la cantidad de batches para pruebas rapidas.

### `train_mode(model)`

Context manager que pone el modelo en modo entrenamiento durante un bloque y luego restaura el modo anterior.

Se usa para calcular loss de validacion en detectores de `torchvision`, porque estos modelos devuelven losses con `targets` solamente cuando estan en modo `train()`.

### `evaluate_detection_loss(model, dataloader, device, max_batches=None)`

Calcula la loss de validacion sin gradientes.

Usa `train_mode(model)` junto con `torch.no_grad()` para obtener el diccionario de losses sin construir grafo ni actualizar pesos. Al terminar restaura el estado original del modelo.

Devuelve:

- `val_loss`
- `val_steps`

## Checkpoints

### `_checkpoint_payload(model, optimizer, scheduler, history, epoch, experiment_name, config)`

Arma el diccionario que se guarda como checkpoint.

Incluye:

- nombre del experimento
- epoca
- pesos del modelo
- estado del optimizador
- historial acumulado
- configuracion
- estado del scheduler si existe

## Orquestacion completa

### `run_detection_experiment(model, train_loader, val_loader, optimizer, device, num_epochs, experiment_name, config, output_dir=None, max_train_batches=None, max_val_batches=None, class_metrics=True, verbose=True)`

Es la funcion principal de entrenamiento end-to-end.

Responsabilidades:

1. mover el modelo al dispositivo
2. entrenar por epoca con `train_one_epoch`
3. evaluar loss de validacion con `evaluate_detection_loss`
4. evaluar mAP con `evaluate_map`
5. extraer metricas principales con `extract_main_map_metrics`
6. mantener historial por epoca
7. detectar el mejor checkpoint por `map`
8. guardar el mejor checkpoint si `output_dir` fue definido
9. devolver un resumen final del experimento

Devuelve un diccionario con:

- `history`
- `best_metric`
- `best_epoch`
- `best_checkpoint_path`
- `best_payload`
- `training_start_time`
- `training_end_time`
- `training_duration_seconds`

### Criterio de mejor modelo

El mejor checkpoint se define por el valor de `map` del conjunto de validacion.

## Restauracion

### `load_checkpoint(model, checkpoint_path, device="cpu")`

Carga un checkpoint desde disco con `torch.load`, restaura `model_state_dict` en el modelo recibido y devuelve el checkpoint completo.

Como los checkpoints de este proyecto guardan un diccionario amplio con:

- `model_state_dict`
- `optimizer_state_dict`
- `history`
- `config`

la funcion fuerza `weights_only=False`.

Esto evita el error introducido por PyTorch 2.6+, donde `torch.load(...)` paso a usar `weights_only=True` por defecto y deja de poder deserializar este formato de checkpoint del proyecto.

## Como se usa en el proyecto

Este modulo permite sacar del notebook la logica repetible de entrenamiento y dejarlo enfocado en configuracion, comparacion de experimentos y analisis de resultados.
