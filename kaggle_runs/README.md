# Kaggle Runs - Entrenamiento de Modelos

Este directorio contiene la configuración y scripts para desplegar, ejecutar y monitorear entrenamientos de modelos de lentes gravitacionales en GPU/TPU usando la plataforma Kaggle.

## Archivos Principales
- **`kernel-metadata.json`**: Archivo de metadatos para la CLI de Kaggle (ID del kernel, conjuntos de datos asociados, activación de GPU e internet).
- **`run_training.sh`**: Script para automatizar el envío del kernel, monitorear el progreso en tiempo real y descargar las salidas.

## Uso

Para enviar y monitorear la ejecución en Kaggle desde la raíz del proyecto:

```bash
./run_training.sh -k jordiancona/lenses-training -p ./kaggle_runs -o ./kaggle_runs/output
```

### Parámetros Principales
- `-k, --kernel-id`: Slug del kernel en Kaggle (`usuario/kernel-slug`).
- `-p, --path-dir`: Directorio que contiene `kernel-metadata.json` y el notebook.
- `-o, --output-dir`: Ruta local donde se guardarán los resultados (default: `./outputs`).
- `-i, --interval`: Intervalo de tiempo en segundos para consultar el estado (default: `30`).

## Salidas Generadas (`./output`)

Al completar el entrenamiento, los resultados se descargan automáticamente en la carpeta de salida:
- **`*.keras`**: Pesos y modelo guardado por cada fold.
- **`*.csv`**: Historial de pérdida y métricas por época (`loss`, `mae`).
- **`*.png`**: Gráficas de curvas de aprendizaje por fold.
- **`*.log`**: Registro de consola de la sesión en Kaggle.
