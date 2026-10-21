import os
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tensorflow as tf

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if os.path.basename(SCRIPT_DIR) == 'training_dataset':
    DEFAULT_TFRECORD_DIR = os.path.join(SCRIPT_DIR, 'tfrecords')
    DEFAULT_PNG_DIR = os.path.join(SCRIPT_DIR, 'tfrecords_pngs')
else:
    DEFAULT_TFRECORD_DIR = os.path.join(SCRIPT_DIR, 'training_dataset', 'tfrecords')
    DEFAULT_PNG_DIR = os.path.join(SCRIPT_DIR, 'training_dataset', 'tfrecords_pngs')

# Terminal colors
CYAN = '\033[36m'
YELLOW = '\033[33m'
GREEN = '\033[32m'
RED = '\033[31m'
BOLD = '\033[1m'
ENDC = '\033[0m'

def parse_tfrecord_with_metadata(example_proto):
    """
    Parse TFRecord protobuf and extract data channels (tensor_4ch, decals_3ch, hst_1ch), physical parameters, and telescope metadata.
    """
    feature_desc = {
        'tensor_4ch': tf.io.FixedLenFeature([], tf.string, default_value=''),
        'shape_4ch': tf.io.VarLenFeature(tf.int64),
        'decals_3ch': tf.io.FixedLenFeature([], tf.string, default_value=''),
        'shape_3ch': tf.io.VarLenFeature(tf.int64),
        'hst_1ch': tf.io.FixedLenFeature([], tf.string, default_value=''),
        'shape_1ch': tf.io.VarLenFeature(tf.int64),
        'name': tf.io.FixedLenFeature([], tf.string, default_value=''),
        'catalog': tf.io.FixedLenFeature([], tf.string, default_value=''),
        'space_telescope': tf.io.FixedLenFeature([], tf.string, default_value='HST'),
        'ra': tf.io.FixedLenFeature([], tf.float32, default_value=0.0),
        'dec': tf.io.FixedLenFeature([], tf.float32, default_value=0.0),
        'z_lens': tf.io.FixedLenFeature([], tf.float32, default_value=-1.0),
        'z_source': tf.io.FixedLenFeature([], tf.float32, default_value=-1.0),
        'theta_E': tf.io.FixedLenFeature([], tf.float32, default_value=-1.0),
        'e1_l': tf.io.FixedLenFeature([], tf.float32, default_value=-1.0),
        'e2_l': tf.io.FixedLenFeature([], tf.float32, default_value=-1.0),
        'grading': tf.io.FixedLenFeature([], tf.string, default_value=''),
        'type': tf.io.FixedLenFeature([], tf.string, default_value=''),
        'ref': tf.io.FixedLenFeature([], tf.string, default_value=''),
        'label': tf.io.FixedLenFeature([], tf.int64, default_value=1),
    }
    parsed = tf.io.parse_single_example(example_proto, feature_desc)
    
    name = parsed['name'].numpy().decode('utf-8')
    catalog = parsed['catalog'].numpy().decode('utf-8')
    space_telescope = parsed['space_telescope'].numpy().decode('utf-8')
    if not space_telescope or space_telescope == 'NONE':
        space_telescope = 'HST'
    ra = float(parsed['ra'].numpy())
    dec = float(parsed['dec'].numpy())
    z_lens = float(parsed['z_lens'].numpy())
    z_source = float(parsed['z_source'].numpy())
    theta_E = float(parsed['theta_E'].numpy())
    e1_l = float(parsed['e1_l'].numpy())
    e2_l = float(parsed['e2_l'].numpy())
    grading = parsed['grading'].numpy().decode('utf-8')
    lens_type = parsed['type'].numpy().decode('utf-8')
    ref = parsed['ref'].numpy().decode('utf-8')
    label = int(parsed['label'].numpy())

    def get_shape(key, default_val):
        if key in parsed and isinstance(parsed[key], tf.SparseTensor):
            vals = parsed[key].values.numpy()
            if len(vals) == 3:
                return list(vals)
        return default_val
    
    tensor_4ch = None
    if len(parsed['tensor_4ch'].numpy()) > 0:
        raw = tf.io.decode_raw(parsed['tensor_4ch'], tf.float32)
        tensor_4ch = tf.reshape(raw, get_shape('shape_4ch', [4, 200, 200])).numpy()

    decals_3ch = None
    if len(parsed['decals_3ch'].numpy()) > 0:
        raw = tf.io.decode_raw(parsed['decals_3ch'], tf.float32)
        decals_3ch = tf.reshape(raw, get_shape('shape_3ch', [3, 200, 200])).numpy()
        
    hst_1ch = None
    if len(parsed['hst_1ch'].numpy()) > 0:
        raw = tf.io.decode_raw(parsed['hst_1ch'], tf.float32)
        hst_1ch = tf.reshape(raw, get_shape('shape_1ch', [1, 200, 200])).numpy()

    if tensor_4ch is not None:
        decals_3ch = tensor_4ch[:3]
        hst_1ch = tensor_4ch[3:4]
    elif decals_3ch is not None and hst_1ch is not None:
        tensor_4ch = np.concatenate([decals_3ch, hst_1ch], axis=0)

    meta = {
        'name': name,
        'catalog': catalog,
        'space_telescope': space_telescope,
        'ra': ra,
        'dec': dec,
        'z_lens': z_lens if z_lens > 0 else None,
        'z_source': z_source if z_source > 0 else None,
        'theta_E': theta_E if theta_E > 0 else None,
        'e1_l': e1_l if e1_l != -1.0 else None,
        'e2_l': e2_l if e2_l != -1.0 else None,
        'grading': grading if grading else None,
        'type': lens_type if lens_type else None,
        'ref': ref if ref else None,
        'label': label
    }
    
    return tensor_4ch, decals_3ch, hst_1ch, meta

def list_available_tfrecords(tfrecord_dir=DEFAULT_TFRECORD_DIR):
    """List all available .tfrecord files in directory, including catalog subdirectories."""
    files = sorted(tf.io.gfile.glob(os.path.join(tfrecord_dir, "**", "*.tfrecord")))
    if not files:
        files = sorted(tf.io.gfile.glob(os.path.join(tfrecord_dir, "*.tfrecord")))
    return files

def plot_single_mock(decals_3ch, hst_1ch, meta, tfrecord_path, save_dir=DEFAULT_PNG_DIR):
    """
    Render 3 independent image panels showing DECaLS RGB, DECaLS Viridis, and Space Observatory Viridis (JWST/HST).
    Dynamically handles missing space imagery (e.g. for ground-only catalog objects like DES).
    Ensures identical vertical orientation across all telescope channels.
    Saves PNG into catalog-specific subdirectory within save_dir.
    """
    cat_name = meta.get('catalog', '') if meta else ''
    space_tele = meta.get('space_telescope', 'HST') if meta else 'HST'
    if not space_tele or space_tele == 'NONE':
        space_tele = 'JWST/HST'

    if not cat_name:
        parent = os.path.basename(os.path.dirname(tfrecord_path))
        if parent and parent.lower() not in ['tfrecords', '.']:
            cat_name = parent
        else:
            cat_name = 'GENERAL'
    else:
        cat_name = str(cat_name).upper().split('/')[0].strip()

    target_save_dir = os.path.join(save_dir, cat_name)
    os.makedirs(target_save_dir, exist_ok=True)

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(13.5, 4.5), facecolor='#121212')
    plt.subplots_adjust(wspace=0.15)
    
    # 1. DECaLS RGB (z=R, r=G, g=B)
    if decals_3ch is not None:
        g, r, z = decals_3ch[0], decals_3ch[1], decals_3ch[2]
        rgb = np.stack([z, r, g], axis=-1)
        rgb = np.clip(rgb, 0, 1)
        ax1.imshow(rgb, origin='lower')
        ax1.set_title("DECaLS (Optical RGB)", color='white', fontsize=11, fontweight='bold', pad=8)
    else:
        ax1.set_facecolor('#181818')
        ax1.text(0.5, 0.5, "Sin Datos DECaLS", color='#888888', fontsize=10, ha='center', va='center', weight='bold')
        ax1.set_title("DECaLS (No disp.)", color='#888888', fontsize=11, fontweight='bold', pad=8)
    ax1.axis('off')
    
    # 2. DECaLS Viridis
    if decals_3ch is not None:
        decals_mean = np.mean(decals_3ch, axis=0)
        ax2.imshow(decals_mean, cmap='viridis', origin='lower')
        ax2.set_title("DECaLS (Viridis 3-Band)", color='white', fontsize=11, fontweight='bold', pad=8)
    else:
        ax2.set_facecolor('#181818')
        ax2.text(0.5, 0.5, "Sin Datos DECaLS", color='#888888', fontsize=10, ha='center', va='center', weight='bold')
        ax2.set_title("DECaLS Viridis (No disp.)", color='#888888', fontsize=11, fontweight='bold', pad=8)
    ax2.axis('off')
    
    # 3. Space Observatory Viridis (JWST / HST, aligned with DECaLS orientation)
    if hst_1ch is not None:
        hst = hst_1ch[0] if hst_1ch.ndim == 3 else hst_1ch
        hst_aligned = np.flipud(hst)
        ax3.imshow(hst_aligned, cmap='viridis', origin='lower')
        ax3.set_title(f"{space_tele} (Space Viridis)", color='white', fontsize=11, fontweight='bold', pad=8)
    else:
        ax3.set_facecolor('#181818')
        ax3.text(0.5, 0.5, f"Sin Cobertura {space_tele}\n(No obs. en MAST)", color='#888888', fontsize=10, ha='center', va='center', weight='bold')
        ax3.set_title(f"{space_tele} (No disp.)", color='white', fontsize=11, fontweight='bold', pad=8)
    ax3.axis('off')
    
    # Save PNG into catalog subfolder using exact TFRecord basename
    tfrecord_basename = os.path.splitext(os.path.basename(tfrecord_path))[0]
    output_png = os.path.join(target_save_dir, f"{tfrecord_basename}.png")
    plt.savefig(output_png, dpi=200, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  {GREEN}✓ PNG guardado en:{ENDC} {os.path.abspath(output_png)}")

def main():
    parser = argparse.ArgumentParser(description="Visualizador de archivos TFRecord (DECaLS / HST / Mocks) -> PNG.")
    parser.add_argument('-p', '--path', type=str, default=DEFAULT_TFRECORD_DIR, help='Ruta a un archivo .tfrecord individual o a una carpeta con archivos .tfrecord')
    parser.add_argument('-n', '--name', type=str, default=None, help='Nombre o patrón del mock o lente a visualizar (ej. J0330-0020 o MOCK_0015)')
    parser.add_argument('-i', '--index', type=int, default=None, help='Índice del archivo TFRecord a visualizar (1-based)')
    parser.add_argument('-a', '--all', action='store_true', help='Convertir TODOS los archivos TFRecord encontrados en la carpeta a PNG')
    parser.add_argument('-l', '--list', action='store_true', help='Listar todos los archivos TFRecord disponibles en la carpeta')
    parser.add_argument('--outdir', type=str, default=DEFAULT_PNG_DIR, help='Directorio de salida para los PNGs')
    
    args = parser.parse_args()
    
    print(f"{BOLD}{CYAN}======================================================{ENDC}")
    print(f"{BOLD}{CYAN}  CONVERTIDOR / VISUALIZADOR DE TFRECORDS A PNG {ENDC}")
    print(f"{BOLD}{CYAN}======================================================{ENDC}\n")

    target_files = []

    if os.path.isfile(args.path):
        target_files = [args.path]

    elif os.path.isdir(args.path):
        all_files = list_available_tfrecords(args.path)
        
        if args.list:
            print(f"{YELLOW}Archivos TFRecord disponibles en '{args.path}' ({len(all_files)} total):{ENDC}")
            for idx, f in enumerate(all_files, 1):
                print(f"  [{idx}] {os.path.relpath(f, args.path)}")
            return

        if args.name is not None:
            target_name = args.name.upper()
            target_files = [f for f in all_files if target_name in f.upper()]
            if not target_files:
                print(f"{RED}No se encontró ningún TFRecord matching con el nombre '{args.name}' en {args.path}{ENDC}")
                return

        elif args.index is not None:
            if 1 <= args.index <= len(all_files):
                target_files = [all_files[args.index - 1]]
            else:
                print(f"{RED}Índice {args.index} fuera de rango. Rango válido: [1 - {len(all_files)}]{ENDC}")
                return

        elif args.all:
            target_files = all_files

        else:
            target_files = all_files
    else:
        print(f"{RED}Ruta no válida: {args.path}{ENDC}")
        return

    if not target_files:
        print(f"{RED}No se encontraron archivos .tfrecord para procesar.{ENDC}")
        return

    print(f"{CYAN}Procesando {len(target_files)} archivo(s) TFRecord y guardando en '{args.outdir}'...{ENDC}\n")
    
    for tfrecord_file in target_files:
        try:
            raw_dataset = tf.data.TFRecordDataset([tfrecord_file])
            for proto in raw_dataset.take(1):
                tensor_4ch, decals_3ch, hst_1ch, meta = parse_tfrecord_with_metadata(proto)
                if decals_3ch is not None or hst_1ch is not None or tensor_4ch is not None:
                    plot_single_mock(decals_3ch, hst_1ch, meta, tfrecord_file, save_dir=args.outdir)
                else:
                    print(f"  {RED}[Error]{ENDC} No se pudieron extraer datos de imagen de {os.path.basename(tfrecord_file)}")
        except Exception as e:
            print(f"  {RED}[Error]{ENDC} Error al procesar {os.path.basename(tfrecord_file)}: {e}")

    print(f"\n{BOLD}{GREEN}======================================================{ENDC}")
    print(f"{BOLD}{GREEN}  ¡PROCESO COMPLETADO! PNG(s) guardados en: {os.path.abspath(args.outdir)}{ENDC}")
    print(f"{BOLD}{GREEN}======================================================{ENDC}\n")

if __name__ == '__main__':
    main()
