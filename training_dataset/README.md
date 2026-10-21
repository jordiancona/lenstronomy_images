# 🌌 Base de Datos de Entrenamiento: Lentes Gravitacionales (Melo et al. 2025)

Este directorio contiene las herramientas, algoritmos y estructuras de datos para la simulación, descarga, procesamiento y visualización de conjuntos de datos de **Lentes Gravitacionales Fuertes**, basados en la metodología de **Melo et al. (2025)** (*A&A 698, A264*) y la librería integrada **[Lenscat](https://github.com/lenscat/lenscat)** (Catálogo Maestro de 32,838 Lentes Gravitacionales Fuertes).

El sistema combina imágenes ópticas terrestres de **DECam / DECaLS** ($g, r, z$) con observaciones espaciales de alta resolución de **HST ACS/WFC** ($F814W$), creando tensores multicanal de $200 \times 200$ píxeles ($10'' \times 10''$ FOV) y simulaciones sintéticas mediante **Lenstronomy**.

---

## 📁 Estructura del Directorio

```text
training_dataset/
├── README.md                     # Documentación oficial del módulo
├── gui_app.py                    # Aplicación de escritorio nativa (PyQt6)
├── import_images_DECam.py        # Motor CLI de descarga (Lenscat/MAST) y simulación de mocks
├── visualize_tfrecords.py        # Visualizador y convertidor de TFRecords a PNG
├── melo2025_lenses_dataset.csv   # Catálogo e índice maestro de metadatos (CSV)
├── tfrecords/                    # Archivos de dataset en formato TensorFlow (.tfrecord)
├── tensors/                      # Archivos de tensores en formato NumPy (.npy)
└── tfrecords_pngs/               # Vistas previas renderizadas en formato PNG (.png)
```

---

## 🖥️ Interfaz Gráfica de Escritorio (`gui_app.py`)

Para operar de manera interactiva sin escribir comandos manuales, se incluye una aplicación nativa de escritorio construida con **PyQt6**:

```bash
# Desde la raíz del repositorio:
python dataset_gui.py

# O directamente desde este directorio:
python training_dataset/gui_app.py
```

### Funcionalidades de la GUI:
1. **🌌 Base Sintética (Mocks):** Configuración del número de lentes sintéticas a generar con Lenstronomy y su formato de exportación.
2. **🔭 Base Real (Lenscat / DECam / HST):** Selección interactiva de catálogos astronómicos mediante la librería **Lenscat** (`CONFIDENT`, `LENSCAT`, `SLACS`, `BELLS`, `DES`, `SUGOHI`, `SQLS`, `CLASS`), límites de descarga y filtros.
3. **🔬 Visualizador de Ejemplo Real (Panel Dedicado):** Selector desplegable por catálogo (`CONFIDENT`, `LENSCAT`, `DES`, `SLACS`, etc.) para examinar metadatos detallados ($RA, Dec, z_l, z_s$, *grading*, *type*, *ref*) e imágenes de cualquier objeto real descargado.
4. **🖼️ Convertidor TFRecords -> PNG:** Renderizado masivo o filtrado de archivos `.tfrecord` a gráficos comparativos.
5. **📊 Galería de Previsualización:** Explorador visual de alta resolución con indicadores de estado en tiempo real.
6. **🚪 Botón Salir:** Botón rojo de salida con diálogo de confirmación para cerrar limpiamente la aplicación.

---

## 🛠️ Herramientas CLI y Scripts

### 1. `import_images_DECam.py` (Simulación y Descarga con Lenscat)

Motor principal para consultar la librería **Lenscat** / MAST, descargar recortes de imágenes y generar simulaciones sintéticas.

#### A. Simular Lentes Sintéticas (*Mocks* con Lenstronomy):
```bash
python training_dataset/import_images_DECam.py -m --num-mocks 100 --dataset-fmt both
```

#### B. Descargar y Procesar Catálogos Reales mediante Lenscat (ej. CONFIDENT, SLACS o DES):
```bash
# Catálogo de lentes confirmadas en Lenscat:
python training_dataset/import_images_DECam.py -c CONFIDENT -s -d --limit 20 --export-dataset --dataset-fmt both

# Catálogo específico (ej. SLACS o DES):
python training_dataset/import_images_DECam.py -c SLACS -s -d --limit 20 --export-dataset --dataset-fmt both
```

#### Parámetros Principales de `import_images_DECam.py`:
- `-m, --generate-mocks`: Activa la simulación sintética de lentes con Lenstronomy.
- `--num-mocks <N>`: Número de lentes *mock* a generar (default: 10).
- `-c, --catalog <NOMBRE>`: Catálogo o lista de Lenscat (`CONFIDENT`, `LENSCAT`, `SLACS`, `BELLS`, `DES`, `SUGOHI`, `SQLS`, `CLASS`).
- `-s, --search-lenses`: Consulta el catálogo mediante `lenscat` y guarda las coordenadas y metadatos en CSV.
- `-d, --download-images`: Descarga los recortes de imágenes desde Legacy Survey / MAST.
- `--export-dataset`: Exporta los datos formateados según Melo et al. (2025).
- `--dataset-fmt <fmt>`: Formato de salida: `tfrecord`, `npy`, o `both`.

---

### 2. `visualize_tfrecords.py` (Visualizador a PNG)

Parsea archivos `.tfrecord` y genera imágenes PNG comparativas de 4 subpantallas:
1. **DECaLS Optical RGB:** Composición a color $z=R, r=G, g=B$.
2. **DECaLS Viridis:** Promedio de brillo de bandas terrestres.
3. **HST F814W Viridis:** Imagen espacial del telescopio Hubble (o indicador si no hay cobertura en MAST).
4. **Tarjeta de Metadatos:** Coordenadas RA/Dec, redshifts ($z_l, z_s$), radio de Einstein ($\theta_E$), elipticidad ($e_1, e_2$) y especificaciones técnicas.

#### Ejemplos de Uso:
```bash
# Convertir TODOS los TFRecords en la carpeta a PNG:
python training_dataset/visualize_tfrecords.py -a

# Convertir un archivo TFRecord específico:
python training_dataset/visualize_tfrecords.py -p ./training_dataset/tfrecords/lens_DES_J035649-240841_ra59.2044_dec-24.1448.tfrecord

# Filtrar por patrón (ej. solo Mocks o solo DES):
python training_dataset/visualize_tfrecords.py -n MOCK
```

---

### 3. `print_parameters.py` (Imprimir Tabla de Parámetros Físicos)

Herramienta para imprimir en consola y/or exportar a CSV los parámetros físicos ($\theta_E, q, e_1, e_2, z_l, z_s$) de todos los sistemas de lentes del dataset y catálogos.

#### Ejemplos de Uso:
```bash
# Imprimir tabla completa de parámetros de todos los sistemas:
python print_parameters.py

# Filtrar por catálogo (ej. solo SLACS o DES) y limitar a 20 resultados:
python print_parameters.py -c SLACS -l 20

# Exportar tabla completa de parámetros físicos a CSV:
python print_parameters.py --save-csv sistemas_parametros.csv
```

---

## 🔬 Especificaciones Técnicas del Dataset

| Parámetro | Valor / Especificación |
| :--- | :--- |
| **Campo de Visión (FOV)** | $10'' \times 10''$ ($0.05''/\text{píxel}$) |
| **Matriz de Píxeles** | $200 \times 200$ píxeles |
| **Banda 1 (Canal 0)** | DECam / DECaLS Filtro $g$ (477 nm) |
| **Banda 2 (Canal 1)** | DECam / DECaLS Filtro $r$ (641 nm) |
| **Banda 3 (Canal 2)** | DECam / DECaLS Filtro $z$ (926 nm) |
| **Banda 4 (Canal 3)** | HST ACS/WFC Filtro $F814W$ (814 nm) |
| **Modelo de Lente (Sintética)** | Singular Isothermal Ellipsoid (SIE) |
| **Modelo de Fuente (Sintética)** | Sersic Ellipse |
| **Normalización** | `AsinhStretch(a=0.05)` en rango $[0, 1]$ |

---

## 📄 Licencia y Referencias
- **Metodología:** Melo et al. (2025), *Astronomy & Astrophysics*, 698, A264.
- **Catálogos Reales:** VizieR CDS, Legacy Survey DR9/DR10 y Mikulski Archive for Space Telescopes (MAST).
