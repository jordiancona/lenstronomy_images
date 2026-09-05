# Kaggle Runs - Datasets y Entrenamiento de Modelos

Este directorio contiene la configuración y scripts para gestionar bases de datos, desplegar, ejecutar y monitorear entrenamientos de modelos de lentes gravitacionales en GPU/TPU usando la plataforma Kaggle.

## Archivos Principales
- **`kernel-metadata.json`**: Archivo de metadatos para la CLI de Kaggle (ID del kernel, conjuntos de datos asociados, activación de GPU e internet).
- **`dataset-metadata.json`**: Archivo de metadatos para la carga de bases de datos a Kaggle (ID, título, licencias).
- **`run_training.sh`**: Script interactivo para automatizar la carga de datasets, envío de kernels/notebooks, monitoreo en tiempo real y descarga de artefactos.

## Uso Interactivo

Simplemente ejecuta el script sin argumentos para desplegar el menú interactivo:

```bash
./run_training.sh
```

El script preguntará paso a paso:
1. **Selección de Modo**: Subir una base de datos (`Dataset`) o subir un script/notebook (`Kernel`).
2. **Si eliges subir un Dataset**:
   - Solicita el directorio local.
   - Pide/confirma los campos de `dataset-metadata.json` (`id`, `title`, `licenses`).
   - Pregunta si es un **nuevo dataset** (`create`) o una **nueva versión** (`version -m`).
3. **Si eliges subir un Kernel/Script**:
   - Solicita la carpeta del proyecto.
   - Pide/confirma paso a paso todos los valores de `kernel-metadata.json` (`id`, `title`, `code_file`, `language`, `kernel_type`, `is_private`, `enable_gpu`, `enable_tpu`, `enable_internet`, `dataset_sources`, etc.), sugiriendo los valores actuales por defecto.
   - Envía el kernel (`push`), monitorea el estado en tiempo real y descarga las salidas al terminar.

## Uso por Línea de Comandos (Flags)

También puedes especificar el modo y parámetros directamente:

```bash
# Subir o monitorear un kernel directamente
./run_training.sh -m kernel -k jordiancona/lenses-training -p ./kaggle_runs -o ./kaggle_runs/output

# Subir un dataset directamente
./run_training.sh -m dataset -p ./mi_dataset
```

### Parámetros Principales
- `-m, --mode`: Modo de operación (`dataset` o `kernel`).
- `-k, --kernel-id`: Slug del kernel en Kaggle (`usuario/kernel-slug`).
- `-p, --path-dir`: Directorio que contiene el código (`kernel-metadata.json`) o los datos (`dataset-metadata.json`).
- `-o, --output-dir`: Ruta local donde se guardarán los resultados del entrenamiento (default: `./output`).
- `-i, --interval`: Segundos entre revisiones de estado (default: `30`).
- `-y, --yes`: Ejecución no interactiva (usa valores existentes/por defecto).

## Salidas Generadas (`./output`)

Al completar el entrenamiento del kernel, los resultados se descargan automáticamente en la carpeta de salida:
- **`*.keras`**: Pesos y modelo guardado por cada fold.
- **`*.csv`**: Historial de pérdida y métricas por época (`loss`, `mae`).
- **`*.png`**: Gráficas de curvas de aprendizaje por fold.
- **`*.log`**: Registro de consola de la sesión en Kaggle.

