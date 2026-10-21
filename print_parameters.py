import os
import sys
import glob
import pandas as pd
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = SCRIPT_DIR
CATALOGS_DIR = os.path.join(PROJECT_ROOT, 'catalogs')
DATASET_DIR = os.path.join(PROJECT_ROOT, 'training_dataset')
TFRECORD_DIR = os.path.join(DATASET_DIR, 'tfrecords')

CYAN = '\033[36m'
YELLOW = '\033[33m'
GREEN = '\033[32m'
RED = '\033[31m'
BOLD = '\033[1m'
ENDC = '\033[0m'

def load_all_parameters():
    """
    Load physical parameters for all lens systems from catalog CSVs and TFRecords.
    Returns a unified pandas DataFrame.
    """
    rows = []
    
    # 1. Load from Catalogs
    if os.path.exists(CATALOGS_DIR):
        for cat_file in sorted(glob.glob(os.path.join(CATALOGS_DIR, '*_lenses.csv'))):
            cat_name = os.path.basename(cat_file).replace('_lenses.csv', '')
            try:
                df_cat = pd.read_csv(cat_file)
                for _, r in df_cat.iterrows():
                    rows.append({
                        'name': str(r.get('name', '')).strip(),
                        'catalog': cat_name,
                        'source': 'Lenscat Catalog' if 'grading' in r else 'VizieR Catalog',
                        'ra': r.get('ra', np.nan),
                        'dec': r.get('dec', np.nan),
                        'z_lens': r.get('z_lens', np.nan),
                        'z_source': r.get('z_source', np.nan),
                        'theta_E': r.get('theta_E', np.nan),
                        'e1_l': r.get('e1_l', np.nan),
                        'e2_l': r.get('e2_l', np.nan),
                        'grading': r.get('grading', np.nan),
                        'type': r.get('type', np.nan),
                    })
            except Exception as err:
                print(f"{RED}Error al leer {cat_file}: {err}{ENDC}")

    # 2. Load from TFRecords if available
    if os.path.exists(TFRECORD_DIR):
        try:
            import tensorflow as tf
            sys.path.insert(0, DATASET_DIR)
            try:
                from visualize_tfrecords import parse_tfrecord_with_metadata
            except ImportError:
                from training_dataset.visualize_tfrecords import parse_tfrecord_with_metadata

            tfrec_files = sorted(glob.glob(os.path.join(TFRECORD_DIR, '**', '*.tfrecord'), recursive=True))
            for tfrec_path in tfrec_files:
                try:
                    raw_ds = tf.data.TFRecordDataset([tfrec_path])
                    for proto in raw_ds.take(1):
                        _, _, _, meta = parse_tfrecord_with_metadata(proto)
                        rows.append({
                            'name': meta['name'],
                            'catalog': meta['catalog'] if meta['catalog'] else 'TFRecord',
                            'source': 'TFRecord File',
                            'ra': meta['ra'],
                            'dec': meta['dec'],
                            'z_lens': meta['z_lens'] if meta['z_lens'] else np.nan,
                            'z_source': meta['z_source'] if meta['z_source'] else np.nan,
                            'theta_E': meta['theta_E'] if meta['theta_E'] else np.nan,
                            'e1_l': meta['e1_l'] if meta['e1_l'] else np.nan,
                            'e2_l': meta['e2_l'] if meta['e2_l'] else np.nan,
                            'grading': meta['grading'] if meta['grading'] else np.nan,
                            'type': meta['type'] if meta['type'] else np.nan,
                        })
                except Exception:
                    pass
        except Exception:
            pass

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    # Deduplicate by name and catalog keeping non-null physical parameters
    df = df.sort_values(by=['theta_E', 'z_lens'], na_position='last')
    df = df.drop_duplicates(subset=['name', 'catalog'], keep='first').reset_index(drop=True)

    # Compute derived physical parameters (q, |e|, phi_deg)
    q_list, e_mag_list, phi_list = [], [], []
    for _, r in df.iterrows():
        e1, e2 = r['e1_l'], r['e2_l']
        if pd.notna(e1) and pd.notna(e2):
            e_mag = np.sqrt(e1**2 + e2**2)
            q_val = (1.0 - e_mag) / (1.0 + e_mag) if e_mag < 1.0 else np.nan
            phi_val = np.degrees(0.5 * np.arctan2(e2, e1)) % 180
            e_mag_list.append(e_mag)
            q_list.append(q_val)
            phi_list.append(phi_val)
        else:
            e_mag_list.append(np.nan)
            q_list.append(np.nan)
            phi_list.append(np.nan)

    df['q_ratio'] = q_list
    df['e_magnitude'] = e_mag_list
    df['phi_deg'] = phi_list

    return df

def print_summary_table(df, catalog_filter=None, limit=None):
    if df.empty:
        print(f"{RED}No se encontraron parámetros de sistemas.{ENDC}")
        return

    if catalog_filter:
        df = df[df['catalog'].str.upper() == catalog_filter.upper()]

    if limit and limit > 0:
        df = df.head(limit)

    print(f"\n{BOLD}{CYAN}===================================================================================================================================={ENDC}")
    print(f"{BOLD}{CYAN}                                           PARÁMETROS FÍSICOS DE SISTEMAS DE LENTES GRAVITACIONALES                                 {ENDC}")
    print(f"{BOLD}{CYAN}===================================================================================================================================={ENDC}")
    
    header = f"{'Nombre':<22} | {'Catálogo':<8} | {'Clasif.':<9} | {'RA (deg)':<9} | {'DEC (deg)':<9} | {'z_l':<6} | {'z_s':<6} | {'θ_E (\")':<8} | {'q (b/a)':<7} | {'e1':<7} | {'e2':<7}"
    print(f"{BOLD}{YELLOW}{header}{ENDC}")
    print("-" * 132)

    for _, r in df.iterrows():
        name = str(r['name'])[:22]
        cat = str(r['catalog'])[:8]
        grading_str = str(r['grading'])[:9] if pd.notna(r['grading']) else "-"
        ra_str = f"{r['ra']:.4f}" if pd.notna(r['ra']) else "N/A"
        dec_str = f"{r['dec']:.4f}" if pd.notna(r['dec']) else "N/A"
        zl_str = f"{r['z_lens']:.3f}" if pd.notna(r['z_lens']) else "-"
        zs_str = f"{r['z_source']:.3f}" if pd.notna(r['z_source']) else "-"
        te_str = f"{r['theta_E']:.2f}" if pd.notna(r['theta_E']) else "-"
        q_str = f"{r['q_ratio']:.2f}" if pd.notna(r['q_ratio']) else "-"
        e1_str = f"{r['e1_l']:.2f}" if pd.notna(r['e1_l']) else "-"
        e2_str = f"{r['e2_l']:.2f}" if pd.notna(r['e2_l']) else "-"

        print(f"{name:<22} | {cat:<8} | {grading_str:<9} | {ra_str:<9} | {dec_str:<9} | {zl_str:<6} | {zs_str:<6} | {te_str:<8} | {q_str:<7} | {e1_str:<7} | {e2_str:<7}")

    print("-" * 132)
    total_valid_theta = df['theta_E'].notna().sum()
    print(f"{BOLD}{GREEN}Total de sistemas: {len(df)} | Sistemas con Radio de Einstein (θ_E) medido: {total_valid_theta}{ENDC}\n")

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Imprimir tabla de parámetros físicos para todos los sistemas de lentes.")
    parser.add_argument('-c', '--catalog', type=str, default=None, help='Filtrar por catálogo (SLACS, DES, SUGOHI, SQLS, MOCK, etc.)')
    parser.add_argument('-l', '--limit', type=int, default=0, help='Límite de renglones a mostrar (0 = todos)')
    parser.add_argument('--save-csv', type=str, default=None, help='Ruta para guardar la tabla completa de parámetros como CSV')

    args = parser.parse_args()
    df = load_all_parameters()

    if args.save_csv:
        out_csv = args.save_csv
        os.makedirs(os.path.dirname(os.path.abspath(out_csv)), exist_ok=True)
        df.to_csv(out_csv, index=False)
        print(f"{GREEN}Tabla de parámetros guardada exitosamente en: {out_csv}{ENDC}")

    print_summary_table(df, catalog_filter=args.catalog, limit=args.limit)

if __name__ == '__main__':
    main()
