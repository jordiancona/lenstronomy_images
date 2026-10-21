import os
import time
import argparse
import requests
import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord
import astropy.units as u
from astropy.io import fits
from astropy.visualization import ImageNormalize, AsinhStretch, PercentileInterval

# pyrefly: ignore [missing-import]
from astroquery.vizier import Vizier
# pyrefly: ignore [missing-import]
from astroquery.mast import Observations

# Determine default dataset directory based on script location
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if os.path.basename(SCRIPT_DIR) == 'training_dataset':
    DEFAULT_DATASET_DIR = SCRIPT_DIR
    PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
else:
    DEFAULT_DATASET_DIR = os.path.join(SCRIPT_DIR, 'training_dataset')
    PROJECT_ROOT = SCRIPT_DIR

# Terminal colors
CYAN = '\033[36m'
YELLOW = '\033[33m'
GREEN = '\033[32m'
RED = '\033[31m'
BOLD = '\033[1m'
ENDC = '\033[0m'

# Catalog mappings in VizieR
# Catalog mappings in Lenscat
CATALOGS = {
    'SLACS': 'Lenscat / SLACS (Bolton et al. 2008 / Auger et al. 2009)',
    'BELLS': 'Lenscat / BELLS (Brownstein et al. 2012 / Shu et al. 2016)',
    'DES': 'Lenscat / DES (Rojas et al. 2022 / Diehl et al. 2017)',
    'SUGOHI': 'Lenscat / SuGOHI (Jaelani et al. 2020 / Sonnenfeld et al. 2018)',
    'SQLS': 'Lenscat / SQLS (Oguri et al. 2006 / Inada et al. 2012)',
    'CLASS': 'Lenscat / CLASS (Myers et al. 2003 / Browne et al. 2003)',
    'CONFIDENT': 'Lenscat / Confident Lenses Master List',
    'LENSCAT': 'Lenscat / Master Catalog (32,838 Lenses)'
}

def clean_lenscat_name(raw_name):
    if not isinstance(raw_name, str):
        return str(raw_name)
    parts = [p.strip() for p in raw_name.split('|') if p.strip() and p.strip() != '-']
    if parts:
        return parts[0]
    return raw_name.strip()

def fetch_vizier_catalog(catalog_key, save_dir=None):
    """Fetch strong gravitational lens catalog using the Lenscat library and save as CSV."""
    if save_dir is None:
        save_dir = os.path.join(PROJECT_ROOT, 'catalogs')
    cat_upper = catalog_key.upper()

    print(f"{CYAN}Consultando catálogo '{cat_upper}' mediante la librería Lenscat...{ENDC}")
    try:
        import lenscat
        df_cat = lenscat.catalog.to_pandas()
        df_cat['clean_name'] = df_cat['name'].apply(clean_lenscat_name)

        if cat_upper in ['SLACS', 'BELLS', 'DES', 'SUGOHI', 'SQLS', 'CLASS']:
            mask = (
                df_cat['clean_name'].str.upper().str.contains(cat_upper) |
                df_cat['name'].str.upper().str.contains(cat_upper) |
                df_cat['ref'].str.upper().str.contains(cat_upper)
            )
            if cat_upper == 'SLACS':
                mask = mask | df_cat['ref'].str.contains('Bolton|Auger', case=False)
            elif cat_upper == 'BELLS':
                mask = mask | df_cat['ref'].str.contains('Shu|Brownstein', case=False)
            elif cat_upper == 'DES':
                mask = mask | df_cat['ref'].str.contains('Rojas|Diehl', case=False)
            elif cat_upper == 'SUGOHI':
                mask = mask | df_cat['ref'].str.contains('Jaelani|Sonnenfeld', case=False)
            elif cat_upper == 'SQLS':
                mask = mask | df_cat['ref'].str.contains('Oguri|Inada', case=False)

            df_filtered = df_cat[mask].copy()

        elif cat_upper in ['CONFIDENT', 'CONFIDENT_LENSCAT']:
            df_filtered = df_cat[df_cat['grading'] == 'confident'].copy()

        else:
            if cat_upper in ['ALL', 'LENSCAT', 'LENSCAT_ALL', 'MASTER']:
                df_filtered = df_cat.copy()
            else:
                mask = (
                    df_cat['clean_name'].str.upper().str.contains(cat_upper) |
                    df_cat['ref'].str.upper().str.contains(cat_upper)
                )
                df_filtered = df_cat[mask].copy()

        if df_filtered.empty:
            print(f"{YELLOW}Advertencia: Lenscat no devolvió resultados específicos para {cat_upper}. Retornando catálogo completo Lenscat...{ENDC}")
            df_filtered = df_cat.copy()

        clean_ras, clean_decs = [], []
        for r, d in zip(df_filtered['RA'], df_filtered['DEC']):
            try: clean_ras.append(float(r))
            except: clean_ras.append(None)
            try: clean_decs.append(float(d))
            except: clean_decs.append(None)

        df = pd.DataFrame({
            'name': df_filtered['clean_name'].values,
            'ra': clean_ras,
            'dec': clean_decs,
            'z_lens': pd.to_numeric(df_filtered['zlens'], errors='coerce').values,
            'grading': df_filtered['grading'].values,
            'type': df_filtered['type'].values,
            'ref': df_filtered['ref'].values,
            'catalog': [cat_upper] * len(df_filtered)
        }).dropna(subset=['ra', 'dec'])

        # Crossmatch physical model parameters (theta_E, e1_l, e2_l, z_source) from local catalog lookup tables
        try:
            phys_rows = []
            for cat_file in glob.glob(os.path.join(save_dir, '*_lenses.csv')):
                try:
                    df_p = pd.read_csv(cat_file)
                    for _, r in df_p.iterrows():
                        if pd.notna(r.get('ra')) and pd.notna(r.get('dec')) and pd.notna(r.get('theta_E')):
                            phys_rows.append({
                                'name': str(r.get('name', '')),
                                'ra': float(r['ra']),
                                'dec': float(r['dec']),
                                'z_source': r.get('z_source', np.nan),
                                'theta_E': float(r['theta_E']),
                                'e1_l': r.get('e1_l', np.nan),
                                'e2_l': r.get('e2_l', np.nan),
                            })
                except Exception:
                    pass

            if phys_rows and len(df) > 0:
                df_phys = pd.DataFrame(phys_rows).drop_duplicates(subset=['ra', 'dec'])
                cat_coords = SkyCoord(ra=df['ra'].values*u.deg, dec=df['dec'].values*u.deg)
                phys_coords = SkyCoord(ra=df_phys['ra'].values*u.deg, dec=df_phys['dec'].values*u.deg)
                
                idx, d2d, _ = cat_coords.match_to_catalog_sky(phys_coords)
                matched_mask = d2d < 3.0 * u.arcsec

                theta_E_arr = np.full(len(df), np.nan)
                z_src_arr = np.full(len(df), np.nan)
                e1_arr = np.full(len(df), np.nan)
                e2_arr = np.full(len(df), np.nan)

                matched_indices = np.where(matched_mask)[0]
                for m_idx in matched_indices:
                    p_idx = idx[m_idx]
                    p_row = df_phys.iloc[p_idx]
                    theta_E_arr[m_idx] = p_row['theta_E']
                    z_src_arr[m_idx] = p_row['z_source']
                    e1_arr[m_idx] = p_row['e1_l']
                    e2_arr[m_idx] = p_row['e2_l']

                df['theta_E'] = theta_E_arr
                df['z_source'] = z_src_arr
                df['e1_l'] = e1_arr
                df['e2_l'] = e2_arr
        except Exception:
            pass

        os.makedirs(save_dir, exist_ok=True)
        csv_path = os.path.join(save_dir, f"{cat_upper}_lenses.csv")
        df.to_csv(csv_path, index=False)

        print(f"{GREEN}¡Catálogo {cat_upper} obtenido exitosamente mediante Lenscat! ({len(df)} lentes){ENDC}")
        print(f"{YELLOW}Guardado en: {csv_path}{ENDC}")
        return df

    except Exception as e:
        print(f"{RED}Error al consultar Lenscat: {e}{ENDC}")
        return None

fetch_lenscat_catalog = fetch_vizier_catalog

def download_decam_image(ra, dec, name, output_dir, pixscale=0.262, size=256, layer='ls-dr9', fmt='both', fov_arcsec=None):
    """Download JPEG and/or FITS cutouts from DECam Legacy Survey and save Viridis PNG false-color image."""
    os.makedirs(output_dir, exist_ok=True)
    
    if fov_arcsec is not None and fov_arcsec > 0:
        pixscale = fov_arcsec / float(size)
    
    jpg_path = os.path.join(output_dir, f"decals_{name}_ra{ra:.4f}_dec{dec:.4f}.jpg")
    fits_path = os.path.join(output_dir, f"decals_{name}_ra{ra:.4f}_dec{dec:.4f}.fits")
    viridis_path = os.path.join(output_dir, f"decals_viridis_{name}_ra{ra:.4f}_dec{dec:.4f}.png")
    
    if fmt in ['jpeg', 'jpg', 'both']:
        jpg_url = f"https://www.legacysurvey.org/viewer/jpeg-cutout?ra={ra}&dec={dec}&pixscale={pixscale}&size={size}&layer={layer}"
        for attempt in range(3):
            try:
                r = requests.get(jpg_url, timeout=15)
                if r.status_code == 200:
                    with open(jpg_path, 'wb') as f:
                        f.write(r.content)
                    print(f"  {GREEN}[DECam JPG]{ENDC} Guardado: {jpg_path}")
                    break
                elif r.status_code == 429:
                    time.sleep(1.0 * (attempt + 1))
            except Exception as e:
                if attempt == 2:
                    print(f"  {RED}[DECam JPG]{ENDC} Error al descargar: {e}")

    if fmt in ['fits', 'both']:
        fits_url = f"https://www.legacysurvey.org/viewer/fits-cutout?ra={ra}&dec={dec}&pixscale={pixscale}&size={size}&layer={layer}"
        for attempt in range(3):
            try:
                r = requests.get(fits_url, timeout=20)
                if r.status_code == 200:
                    with open(fits_path, 'wb') as f:
                        f.write(r.content)
                    print(f"  {GREEN}[DECam FITS]{ENDC} Guardado: {fits_path}")
                    
                    try:
                        import matplotlib
                        matplotlib.use('Agg')
                        import matplotlib.pyplot as plt
                        
                        with fits.open(fits_path) as hdul:
                            data = hdul[0].data
                            if data is not None:
                                mean_img = np.mean(data, axis=0) if data.ndim == 3 else data
                                mean_img = np.nan_to_num(mean_img)
                                norm = ImageNormalize(mean_img, interval=PercentileInterval(99.5), stretch=AsinhStretch(a=0.05))
                                norm_data = norm(mean_img)
                                
                                fig, ax = plt.subplots(figsize=(6, 6))
                                ax.imshow(norm_data, cmap='viridis', origin='lower')
                                ax.axis('off')
                                plt.savefig(viridis_path, bbox_inches='tight', pad_inches=0, dpi=200)
                                plt.close(fig)
                                print(f"  {GREEN}[DECam Viridis]{ENDC} PNG guardado: {viridis_path}")
                    except Exception as err:
                        print(f"  {YELLOW}[DECam Viridis]{ENDC} No se pudo generar PNG Viridis: {err}")
                    break
                elif r.status_code == 429:
                    time.sleep(1.0 * (attempt + 1))
            except Exception as e:
                if attempt == 2:
                    print(f"  {RED}[DECam FITS]{ENDC} Error al descargar: {e}")

def download_space_image(ra, dec, name, output_dir, radius=5.0, filter_name='F814W'):
    """
    Query MAST for Space Observatory imagery (prioritizing JWST, falling back to HST ACS/WFC).
    Downloads FITS data and saves a high-contrast Viridis PNG cutout.
    Returns (fits_path, space_telescope_name).
    """
    os.makedirs(output_dir, exist_ok=True)
    fits_path = os.path.join(output_dir, f"space_{name}_ra{ra:.4f}_dec{dec:.4f}.fits")
    legacy_hst_fits_path = os.path.join(output_dir, f"hst_{name}_ra{ra:.4f}_dec{dec:.4f}_{filter_name}.fits")
    viridis_path = os.path.join(output_dir, f"space_viridis_{name}_ra{ra:.4f}_dec{dec:.4f}.png")
    legacy_viridis_path = os.path.join(output_dir, f"hst_viridis_{name}_ra{ra:.4f}_dec{dec:.4f}.png")
    
    try:
        coord = SkyCoord(ra=ra*u.deg, dec=dec*u.deg)
        space_telescope = "JWST"
        
        # 1. Search MAST for JWST high-resolution space observations
        obs = Observations.query_criteria(
            coordinates=coord,
            radius=radius * u.arcsec,
            obs_collection="JWST",
            dataproduct_type="image"
        )
        
        # 2. If no JWST observations, fall back to HST
        if len(obs) == 0:
            space_telescope = "HST"
            obs = Observations.query_criteria(
                coordinates=coord,
                radius=radius * u.arcsec,
                obs_collection="HST",
                dataproduct_type="image",
                filters=filter_name
            )
            if len(obs) == 0:
                obs = Observations.query_criteria(
                    coordinates=coord,
                    radius=radius * u.arcsec,
                    obs_collection="HST",
                    dataproduct_type="image"
                )
            
        if len(obs) == 0:
            print(f"  {YELLOW}[MAST Espacial]{ENDC} No se encontraron observaciones (JWST/HST) en MAST para {name}")
            return None, "NONE"
            
        products = Observations.get_product_list(obs[0])
        filtered_p = Observations.filter_products(
            products,
            productSubGroupDescription=["DRC", "DRZ", "FLT", "I2D", "CALIMAGE", "CAL", "TRAPSFILLED"],
            extension="fits"
        )
        if len(filtered_p) == 0:
            filtered_p = Observations.filter_products(products, extension="fits")
            
        if len(filtered_p) == 0:
            print(f"  {YELLOW}[MAST {space_telescope}]{ENDC} No se encontraron archivos FITS válidos en MAST para {name}")
            return None, "NONE"
            
        print(f"  {CYAN}[MAST {space_telescope}]{ENDC} Descargando FITS ({space_telescope}) desde MAST para {name}...")
        download = Observations.download_products(filtered_p[0:1], download_dir=output_dir)
        
        if len(download) > 0:
            downloaded_fits = download['Local Path'][0]
            if os.path.exists(fits_path):
                os.remove(fits_path)
            os.rename(downloaded_fits, fits_path)
            
            try:
                import shutil
                shutil.copyfile(fits_path, legacy_hst_fits_path)
            except Exception:
                pass

            print(f"  {GREEN}[{space_telescope} FITS]{ENDC} Guardado: {fits_path}")
            
            try:
                import matplotlib
                matplotlib.use('Agg')
                import matplotlib.pyplot as plt
                
                with fits.open(fits_path) as hdul:
                    data = None
                    for hdu in hdul:
                        if hdu.data is not None and hdu.data.ndim == 2:
                            data = hdu.data
                            break
                    if data is not None:
                        data = np.nan_to_num(data)
                        norm = ImageNormalize(data, interval=PercentileInterval(99.5), stretch=AsinhStretch(a=0.05))
                        res = norm(data)
                        if isinstance(res, np.ma.MaskedArray):
                            res = res.filled(0.0)
                        norm_data = np.asarray(res, dtype=np.float32)
                        
                        fig, ax = plt.subplots(figsize=(6, 6))
                        ax.imshow(norm_data, cmap='viridis', origin='lower')
                        ax.axis('off')
                        plt.savefig(viridis_path, bbox_inches='tight', pad_inches=0, dpi=200)
                        plt.close(fig)
                        
                        try:
                            import shutil
                            shutil.copyfile(viridis_path, legacy_viridis_path)
                        except Exception:
                            pass

                        print(f"  {GREEN}[{space_telescope} Viridis]{ENDC} PNG guardado: {viridis_path}")
            except Exception as err:
                print(f"  {YELLOW}[{space_telescope} Viridis]{ENDC} No se pudo generar PNG Viridis: {err}")
                
            return fits_path, space_telescope
            
    except Exception as e:
        print(f"  {RED}[MAST Espacial]{ENDC} Error al descargar de MAST para {name}: {e}")
        return None, "NONE"

def download_hst_image(ra, dec, name, output_dir, radius=5.0, filter_name='F814W'):
    """Backward compatibility wrapper for download_space_image."""
    fits_p, _ = download_space_image(ra, dec, name, output_dir, radius=radius, filter_name=filter_name)
    return fits_p

def save_comparison_plot(decam_jpg, decam_vir, hst_png, output_path, space_telescope="Space"):
    """Generate side-by-side comparison plot of available DECam and Space (JWST/HST) images."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        
        images, titles = [], []
        if os.path.exists(decam_jpg):
            images.append(plt.imread(decam_jpg))
            titles.append("DECam (Optical RGB)")
        if os.path.exists(decam_vir):
            images.append(plt.imread(decam_vir))
            titles.append("DECam (Viridis 3-Band)")
        if os.path.exists(hst_png):
            images.append(plt.imread(hst_png))
            titles.append(f"{space_telescope} (Space Viridis)")
            
        if not images:
            return
            
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        fig, axes = plt.subplots(1, len(images), figsize=(5 * len(images), 5))
        if len(images) == 1:
            axes = [axes]
            
        for ax, img, title in zip(axes, images, titles):
            ax.imshow(img)
            ax.set_title(title, fontsize=12, pad=10, fontweight='bold')
            ax.axis('off')
            
        plt.tight_layout()
        plt.savefig(output_path, dpi=200, bbox_inches='tight')
        plt.close(fig)
        print(f"  {GREEN}[Comparación]{ENDC} Guardada: {output_path}")
    except Exception as e:
        print(f"  {RED}[Comparación]{ENDC} Error al generar gráfica comparativa: {e}")

def export_melo2025_dataset_tensor(decam_fits_path, hst_fits_path, ra, dec, name, catalog_name="SLACS", dataset_dir=DEFAULT_DATASET_DIR, dataset_fmt="both", z_lens=None, z_source=None, theta_E=None, e1_l=None, e2_l=None, grading=None, type_str=None, ref=None, space_telescope="HST"):
    """
    Process DECam FITS and HST FITS, extract cutouts, normalize [0, 1],
    and construct 4-channel array matching Melo et al. (2025) (A&A 698, A264).
    """
    try:
        os.makedirs(os.path.join(dataset_dir, "tensors"), exist_ok=True)
        os.makedirs(os.path.join(dataset_dir, "tfrecords"), exist_ok=True)
        clean_name = str(name).strip().replace(' ', '_').replace('/', '_')

        decam_data = None
        if os.path.exists(decam_fits_path):
            with fits.open(decam_fits_path) as hdul:
                decam_data = hdul[0].data

        hst_data = None
        if hst_fits_path and os.path.exists(hst_fits_path):
            with fits.open(hst_fits_path) as hdul:
                for hdu in hdul:
                    if hdu.data is not None and hdu.data.ndim == 2:
                        hst_data = hdu.data
                        break

        if decam_data is None and hst_data is None:
            return False

        target_shape = (200, 200)

        def normalize_and_crop(arr, is_3d=False):
            if arr is None:
                return None
            arr = np.nan_to_num(arr)

            if is_3d:
                if arr.ndim == 2:
                    arr = np.stack([arr, arr, arr], axis=0)
                c_y, c_x = arr.shape[1] // 2, arr.shape[2] // 2
                h_y, h_x = target_shape[0] // 2, target_shape[1] // 2
                start_y = max(0, c_y - h_y)
                end_y = start_y + target_shape[0]
                start_x = max(0, c_x - h_x)
                end_x = start_x + target_shape[1]
                crop = arr[:, start_y:end_y, start_x:end_x]

                if crop.shape[1:] != target_shape:
                    padded = np.zeros((3, target_shape[0], target_shape[1]), dtype=np.float32)
                    padded[:, :crop.shape[1], :crop.shape[2]] = crop
                    crop = padded

                norm_channels = []
                for ch in range(3):
                    sub = crop[ch] - np.median(crop[ch])
                    sub[sub < 0] = 0
                    norm = ImageNormalize(sub, interval=PercentileInterval(99.5), stretch=AsinhStretch(a=0.05))
                    res = norm(sub)
                    if isinstance(res, np.ma.MaskedArray):
                        res = res.filled(0.0)
                    norm_channels.append(np.asarray(res, dtype=np.float32))
                return np.ascontiguousarray(np.array(norm_channels, dtype=np.float32))

            else:
                if arr.ndim == 3:
                    arr = np.mean(arr, axis=0)
                c_y, c_x = arr.shape[0] // 2, arr.shape[1] // 2
                h_y, h_x = target_shape[0] // 2, target_shape[1] // 2
                start_y = max(0, c_y - h_y)
                end_y = start_y + target_shape[0]
                start_x = max(0, c_x - h_x)
                end_x = start_x + target_shape[1]
                crop = arr[start_y:end_y, start_x:end_x]

                if crop.shape != target_shape:
                    padded = np.zeros(target_shape, dtype=np.float32)
                    padded[:crop.shape[0], :crop.shape[1]] = crop
                    crop = padded

                sub = crop - np.median(crop)
                sub[sub < 0] = 0
                norm = ImageNormalize(sub, interval=PercentileInterval(99.5), stretch=AsinhStretch(a=0.05))
                res = norm(sub)
                if isinstance(res, np.ma.MaskedArray):
                    res = res.filled(0.0)
                return np.ascontiguousarray(np.asarray(res, dtype=np.float32))

        decam_norm = normalize_and_crop(decam_data, is_3d=True) if decam_data is not None else None
        hst_norm = normalize_and_crop(hst_data, is_3d=False) if hst_data is not None else None

        tensor_4ch = None
        has_decam = decam_norm is not None
        has_hst = hst_norm is not None

        if has_decam and has_hst:
            tensor_4ch = np.zeros((4, target_shape[0], target_shape[1]), dtype=np.float32)
            tensor_4ch[:3] = decam_norm
            tensor_4ch[3] = hst_norm

        cat_subfolder = str(catalog_name).upper()
        tensor_cat_dir = os.path.join(dataset_dir, "tensors", cat_subfolder)
        tfrec_cat_dir = os.path.join(dataset_dir, "tfrecords", cat_subfolder)

        if dataset_fmt in ["npy", "both"]:
            os.makedirs(tensor_cat_dir, exist_ok=True)
            if tensor_4ch is not None:
                tensor_4ch_path = os.path.join(tensor_cat_dir, f"tensor_4ch_{clean_name}_ra{ra:.4f}_dec{dec:.4f}.npy")
                np.save(tensor_4ch_path, tensor_4ch)

            if has_decam:
                decals_3ch_path = os.path.join(tensor_cat_dir, f"decals_3ch_{clean_name}_ra{ra:.4f}_dec{dec:.4f}.npy")
                np.save(decals_3ch_path, decam_norm)

            if has_hst:
                hst_1ch_path = os.path.join(tensor_cat_dir, f"hst_1ch_{clean_name}_ra{ra:.4f}_dec{dec:.4f}.npy")
                np.save(hst_1ch_path, hst_norm)

        tfrecord_path = None
        if dataset_fmt in ["tfrecord", "both"]:
            try:
                os.makedirs(tfrec_cat_dir, exist_ok=True)
                import tensorflow as tf
                tfrecord_path = os.path.join(tfrec_cat_dir, f"lens_{clean_name}_ra{ra:.4f}_dec{dec:.4f}.tfrecord")
                
                features = {
                    'name': tf.train.Feature(bytes_list=tf.train.BytesList(value=[name.encode('utf-8')])),
                    'ra': tf.train.Feature(float_list=tf.train.FloatList(value=[float(ra)])),
                    'dec': tf.train.Feature(float_list=tf.train.FloatList(value=[float(dec)])),
                    'catalog': tf.train.Feature(bytes_list=tf.train.BytesList(value=[catalog_name.encode('utf-8')])),
                    'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[1])),
                }
                
                if z_lens is not None and str(z_lens).strip() != '' and not (isinstance(z_lens, float) and np.isnan(z_lens)):
                    try: features['z_lens'] = tf.train.Feature(float_list=tf.train.FloatList(value=[float(z_lens)]))
                    except: pass
                if z_source is not None and str(z_source).strip() != '' and not (isinstance(z_source, float) and np.isnan(z_source)):
                    try: features['z_source'] = tf.train.Feature(float_list=tf.train.FloatList(value=[float(z_source)]))
                    except: pass
                if theta_E is not None and str(theta_E).strip() != '' and not (isinstance(theta_E, float) and np.isnan(theta_E)):
                    try: features['theta_E'] = tf.train.Feature(float_list=tf.train.FloatList(value=[float(theta_E)]))
                    except: pass
                if e1_l is not None and str(e1_l).strip() != '' and not (isinstance(e1_l, float) and np.isnan(e1_l)):
                    try: features['e1_l'] = tf.train.Feature(float_list=tf.train.FloatList(value=[float(e1_l)]))
                    except: pass
                if e2_l is not None and str(e2_l).strip() != '' and not (isinstance(e2_l, float) and np.isnan(e2_l)):
                    try: features['e2_l'] = tf.train.Feature(float_list=tf.train.FloatList(value=[float(e2_l)]))
                    except: pass

                if grading is not None and str(grading).strip() != '':
                    features['grading'] = tf.train.Feature(bytes_list=tf.train.BytesList(value=[str(grading).encode('utf-8')]))
                if type_str is not None and str(type_str).strip() != '':
                    features['type'] = tf.train.Feature(bytes_list=tf.train.BytesList(value=[str(type_str).encode('utf-8')]))
                if ref is not None and str(ref).strip() != '':
                    features['ref'] = tf.train.Feature(bytes_list=tf.train.BytesList(value=[str(ref).encode('utf-8')]))
                if space_telescope is not None and str(space_telescope).strip() != '':
                    features['space_telescope'] = tf.train.Feature(bytes_list=tf.train.BytesList(value=[str(space_telescope).encode('utf-8')]))

                if tensor_4ch is not None:
                    features['tensor_4ch'] = tf.train.Feature(bytes_list=tf.train.BytesList(value=[tensor_4ch.tobytes()]))
                    features['shape_4ch'] = tf.train.Feature(int64_list=tf.train.Int64List(value=list(tensor_4ch.shape)))
                    
                if has_decam:
                    features['decals_3ch'] = tf.train.Feature(bytes_list=tf.train.BytesList(value=[decam_norm.tobytes()]))
                    features['shape_3ch'] = tf.train.Feature(int64_list=tf.train.Int64List(value=list(decam_norm.shape)))
                    
                if has_hst:
                    features['hst_1ch'] = tf.train.Feature(bytes_list=tf.train.BytesList(value=[hst_norm.tobytes()]))
                    features['shape_1ch'] = tf.train.Feature(int64_list=tf.train.Int64List(value=list(hst_norm.shape)))

                example = tf.train.Example(features=tf.train.Features(feature=features))
                with tf.io.TFRecordWriter(tfrecord_path) as writer:
                    writer.write(example.SerializeToString())
            except Exception as tf_err:
                print(f"  {RED}[TFRecord Export]{ENDC} Error al generar TFRecord para {name}: {tf_err}")

        meta_csv = os.path.join(dataset_dir, "melo2025_lenses_dataset.csv")
        row = {
            'name': name,
            'ra': ra,
            'dec': dec,
            'catalog': catalog_name,
            'z_lens': z_lens if z_lens is not None else '',
            'z_source': z_source if z_source is not None else '',
            'theta_E': theta_E if theta_E is not None else '',
            'e1_l': e1_l if e1_l is not None else '',
            'e2_l': e2_l if e2_l is not None else '',
            'grading': grading if grading is not None else '',
            'type': type_str if type_str is not None else '',
            'ref': ref if ref is not None else '',
            'space_telescope': space_telescope if space_telescope is not None else 'HST',
            'has_decam': has_decam,
            'has_hst': has_hst,
            'label': 1,
            'tensor_4ch_path': os.path.join(dataset_dir, "tensors", f"tensor_4ch_{clean_name}_ra{ra:.4f}_dec{dec:.4f}.npy") if tensor_4ch is not None else '',
            'tfrecord_path': tfrecord_path if tfrecord_path else ''
        }

        if os.path.exists(meta_csv):
            try:
                df_existing = pd.read_csv(meta_csv, on_bad_lines='skip')
                df_existing = df_existing[df_existing['name'] != name]
                df_updated = pd.concat([df_existing, pd.DataFrame([row])], ignore_index=True)
            except Exception:
                df_updated = pd.DataFrame([row])
        else:
            df_updated = pd.DataFrame([row])

        df_updated.to_csv(meta_csv, index=False)
        print(f"  {GREEN}[Dataset Melo 2025]{ENDC} Registro actualizado en CSV: {name} (DECam: {has_decam}, HST: {has_hst})")
        return True
    except Exception as e:
        print(f"  {RED}[Dataset Export]{ENDC} Error al exportar dataset Melo 2025: {e}")
        return False

def generate_mock_lenses(num_mocks=10, dataset_dir=DEFAULT_DATASET_DIR, dataset_fmt="both"):
    """
    Generate mock gravitational lens systems (Melo et al. 2025 approach) by superimposing
    Lenstronomy SIE + Sersic lensed arcs onto 4-channel ground/space grid arrays, saved as TFRecords.
    """
    print(f"\n{CYAN}Simulando {num_mocks} lentes gravitacionales 'mocks' (Melo et al. 2025)...{ENDC}")
    try:
        from lenstronomy.LensModel.lens_model import LensModel
        from lenstronomy.LightModel.light_model import LightModel
        import tensorflow as tf

        os.makedirs(os.path.join(dataset_dir, "tensors"), exist_ok=True)
        os.makedirs(os.path.join(dataset_dir, "tfrecords"), exist_ok=True)

        num_pix = 200
        delta_pix = 0.05

        x, y = np.meshgrid(
            np.linspace(-num_pix / 2 * delta_pix, num_pix / 2 * delta_pix, num_pix),
            np.linspace(-num_pix / 2 * delta_pix, num_pix / 2 * delta_pix, num_pix)
        )

        meta_rows = []

        for i in range(1, num_mocks + 1):
            mock_name = f"MOCK_{i:04d}"
            
            theta_E = float(np.random.uniform(1.0, 2.2))
            q_l = float(np.random.uniform(0.5, 0.9))
            phi_l = float(np.random.uniform(0, np.pi))
            e_l = (1.0 - q_l) / (1.0 + q_l)
            e1_l, e2_l = float(e_l * np.cos(2 * phi_l)), float(e_l * np.sin(2 * phi_l))

            x_s = float(np.random.uniform(-0.25, 0.25))
            y_s = float(np.random.uniform(-0.25, 0.25))
            re_s = float(np.random.uniform(0.12, 0.35))
            q_s = float(np.random.uniform(0.4, 0.9))
            phi_s = float(np.random.uniform(0, np.pi))
            e_s = (1.0 - q_s) / (1.0 + q_s)
            e1_s, e2_s = float(e_s * np.cos(2 * phi_s)), float(e_s * np.sin(2 * phi_s))

            lens_model = LensModel(['SIE'])
            lens_kwargs = [{'theta_E': theta_E, 'e1': e1_l, 'e2': e2_l, 'center_x': 0.0, 'center_y': 0.0}]
            x_lensed, y_lensed = lens_model.ray_shooting(x, y, lens_kwargs)

            lens_light_model = LightModel(['SERSIC_ELLIPSE'])
            lens_light_kwargs = [{'amp': 15.0, 'R_sersic': 0.6, 'n_sersic': 4.0, 'e1': e1_l, 'e2': e2_l, 'center_x': 0.0, 'center_y': 0.0}]
            lens_light = lens_light_model.surface_brightness(x, y, lens_light_kwargs).reshape(num_pix, num_pix)

            source_light_model = LightModel(['SERSIC_ELLIPSE'])
            source_light_kwargs = [{'amp': 45.0, 'R_sersic': re_s, 'n_sersic': 1.5, 'e1': e1_s, 'e2': e2_s, 'center_x': x_s, 'center_y': y_s}]
            source_light = source_light_model.surface_brightness(x_lensed, y_lensed, source_light_kwargs).reshape(num_pix, num_pix)

            noise_decam = np.random.normal(0, 0.05, (3, num_pix, num_pix)).astype(np.float32)
            noise_hst = np.random.normal(0, 0.02, (1, num_pix, num_pix)).astype(np.float32)

            g_light = lens_light * 0.7 + source_light * 0.9 + noise_decam[0]
            r_light = lens_light * 1.0 + source_light * 1.0 + noise_decam[1]
            z_light = lens_light * 1.3 + source_light * 0.7 + noise_decam[2]
            hst_light = lens_light * 1.1 + source_light * 1.2 + noise_hst[0]

            def norm_arr(arr):
                arr = np.nan_to_num(arr)
                bg = np.median(arr)
                arr_sub = arr - bg
                arr_sub[arr_sub < 0] = 0
                norm = ImageNormalize(arr_sub, interval=PercentileInterval(99.5), stretch=AsinhStretch(a=0.05))
                return norm(arr_sub).astype(np.float32)

            decam_norm = np.array([norm_arr(g_light), norm_arr(r_light), norm_arr(z_light)], dtype=np.float32)
            hst_norm = np.array([norm_arr(hst_light)], dtype=np.float32)

            tensor_4ch = np.zeros((4, num_pix, num_pix), dtype=np.float32)
            tensor_4ch[:3] = decam_norm
            tensor_4ch[3] = hst_norm[0]

            tfrec_mock_dir = os.path.join(dataset_dir, "tfrecords", "MOCK")
            tensor_mock_dir = os.path.join(dataset_dir, "tensors", "MOCK")
            os.makedirs(tfrec_mock_dir, exist_ok=True)
            os.makedirs(tensor_mock_dir, exist_ok=True)

            tfrecord_path = os.path.join(tfrec_mock_dir, f"mock_lens_{mock_name}.tfrecord")
            tensor_path = os.path.join(tensor_mock_dir, f"mock_tensor_4ch_{mock_name}.npy")
            
            features = {
                'name': tf.train.Feature(bytes_list=tf.train.BytesList(value=[mock_name.encode('utf-8')])),
                'catalog': tf.train.Feature(bytes_list=tf.train.BytesList(value=[b'MOCK'])),
                'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[1])),
                'tensor_4ch': tf.train.Feature(bytes_list=tf.train.BytesList(value=[tensor_4ch.tobytes()])),
                'shape_4ch': tf.train.Feature(int64_list=tf.train.Int64List(value=list(tensor_4ch.shape))),
                'decals_3ch': tf.train.Feature(bytes_list=tf.train.BytesList(value=[decam_norm.tobytes()])),
                'shape_3ch': tf.train.Feature(int64_list=tf.train.Int64List(value=list(decam_norm.shape))),
                'hst_1ch': tf.train.Feature(bytes_list=tf.train.BytesList(value=[hst_norm.tobytes()])),
                'shape_1ch': tf.train.Feature(int64_list=tf.train.Int64List(value=list(hst_norm.shape))),
                'theta_E': tf.train.Feature(float_list=tf.train.FloatList(value=[theta_E])),
                'e1_l': tf.train.Feature(float_list=tf.train.FloatList(value=[e1_l])),
                'e2_l': tf.train.Feature(float_list=tf.train.FloatList(value=[e2_l])),
            }

            example = tf.train.Example(features=tf.train.Features(feature=features))
            with tf.io.TFRecordWriter(tfrecord_path) as writer:
                writer.write(example.SerializeToString())

            if dataset_fmt in ["npy", "both"]:
                np.save(tensor_path, tensor_4ch)

            meta_rows.append({
                'name': mock_name,
                'ra': 0.0,
                'dec': 0.0,
                'catalog': 'MOCK',
                'z_lens': 0.3,
                'z_source': 1.2,
                'has_decam': True,
                'has_hst': True,
                'label': 1,
                'tensor_4ch_path': tensor_path if dataset_fmt in ["npy", "both"] else '',
                'tfrecord_path': tfrecord_path
            })
            print(f"  {GREEN}[Mock Lens]{ENDC} Generado mock {mock_name} -> {tfrecord_path}")

        meta_csv = os.path.join(dataset_dir, "melo2025_lenses_dataset.csv")
        df_mock = pd.DataFrame(meta_rows)
        if os.path.exists(meta_csv):
            df_existing = pd.read_csv(meta_csv)
            df_existing = df_existing[~df_existing['name'].str.startswith('MOCK_')]
            df_combined = pd.concat([df_existing, df_mock], ignore_index=True)
        else:
            df_combined = df_mock
        df_combined.to_csv(meta_csv, index=False)

        print(f"{GREEN}¡Generación de {num_mocks} mocks completada exitosamente!{ENDC}")
        return True
    except Exception as e:
        print(f"{RED}Error al generar lentes mock: {e}{ENDC}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Herramienta de búsqueda, descarga masiva y simulación de Lentes Gravitacionales (DECam, HST, Mocks).")
    
    parser.add_argument('-c', '--catalog', type=str, default='SLACS', help='Nombre del catálogo (SLACS, BELLS, DES, SUGOHI, SQLS, CLASS)')
    parser.add_argument('-s', '--search-lenses', action='store_true', help='Buscar catálogo en VizieR y guardar como CSV')
    parser.add_argument('-d', '--download-images', action='store_true', help='Descargar imágenes para las lentes')
    parser.add_argument('-m', '--generate-mocks', action='store_true', help='Simular lentes sintéticas (Mocks con Lenstronomy) y guardar en TFRecords')
    parser.add_argument('--num-mocks', type=int, default=10, help='Número de lentes mock a generar (default: 10)')
    
    parser.add_argument('--ra', type=float, default=None, help='RA individual en grados decimales')
    parser.add_argument('--dec', type=float, default=None, help='Dec individual en grados decimales')
    parser.add_argument('--name', type=str, default=None, help='Nombre del objeto individual')
    
    parser.add_argument('--source', type=str, choices=['decam', 'hst', 'both'], default='both', help='Origen de imágenes: decam, hst, o both')
    parser.add_argument('--fmt', '--format', type=str, choices=['jpeg', 'jpg', 'fits', 'both'], default='both', help='Formato de descarga DECam: jpeg, fits, o both')
    parser.add_argument('-l', '--limit', type=int, default=10, help='Límite de imágenes a descargar (0 o negativo para descargar TODO)')
    parser.add_argument('-w', '--workers', type=int, default=4, help='Número de descargas simultáneas en paralelo (default: 4)')
    
    parser.add_argument('--fov', type=float, default=10.0, help='Campo de visión en segundos de arco (default: 10.0 arcsec para 10"x10")')
    parser.add_argument('--pixscale', type=float, default=0.262, help='Escala de píxel en arcsec/px (DECam predeterminado: 0.262)')
    parser.add_argument('--size', type=int, default=256, help='Tamaño del recorte en píxeles (default: 256)')
    parser.add_argument('--layer', type=str, default='ls-dr9', help='Capa de Legacy Survey (ls-dr9, ls-dr10)')
    parser.add_argument('--hst-filter', type=str, default='F814W', help='Filtro preferido de HST (default: F814W)')
    parser.add_argument('-r', '--radius', type=float, default=5.0, help='Radio de búsqueda en MAST (arcsec)')
    parser.add_argument('--outdir', type=str, default=os.path.join(PROJECT_ROOT, 'downloaded_lenses'), help='Directorio base de descarga')
    
    parser.add_argument('--export-dataset', action='store_true', help='Generar dataset estructurado para Deep Learning (Melo et al. 2025)')
    parser.add_argument('--dataset-fmt', type=str, choices=['npy', 'tfrecord', 'both'], default='both', help='Formato del dataset: npy, tfrecord, o both (default: both)')
    parser.add_argument('--dataset-dir', type=str, default=DEFAULT_DATASET_DIR, help='Directorio de salida para la base de datos de entrenamiento')
    
    args = parser.parse_args()
    
    print(f"{BOLD}{CYAN}======================================================{ENDC}")
    print(f"{BOLD}{CYAN}  BUSCADOR Y SIMULADOR DE LENTES GRAVITACIONALES {ENDC}")
    print(f"{BOLD}{CYAN}======================================================{ENDC}\n")

    if args.generate_mocks:
        generate_mock_lenses(num_mocks=args.num_mocks, dataset_dir=args.dataset_dir, dataset_fmt=args.dataset_fmt)

    df_lenses = None
    if args.search_lenses:
        df_lenses = fetch_vizier_catalog(args.catalog)
        
    if args.download_images:
        targets = []
        
        if args.ra is not None and args.dec is not None:
            name = args.name if args.name else f"custom_{args.ra:.4f}_{args.dec:.4f}"
            targets.append({'name': name, 'ra': args.ra, 'dec': args.dec})
            print(f"{CYAN}Modo individual: Descargando para {name} (RA={args.ra}, Dec={args.dec}){ENDC}\n")
            
        else:
            csv_path = os.path.join(PROJECT_ROOT, 'catalogs', f"{args.catalog.upper()}_lenses.csv")
            if os.path.exists(csv_path):
                print(f"{YELLOW}Cargando lista de lentes desde {csv_path}...{ENDC}")
                df_lenses = pd.read_csv(csv_path)
            else:
                print(f"{YELLOW}CSV no encontrado localmente. Consultando VizieR para {args.catalog.upper()}...{ENDC}")
                df_lenses = fetch_vizier_catalog(args.catalog)
                
            if df_lenses is not None and not df_lenses.empty:
                for idx, row in df_lenses.iterrows():
                    target_dict = {
                        'name': str(row['name']),
                        'ra': float(row['ra']),
                        'dec': float(row['dec'])
                    }
                    if 'z_lens' in row and pd.notna(row['z_lens']):
                        target_dict['z_lens'] = row['z_lens']
                    if 'z_source' in row and pd.notna(row['z_source']):
                        target_dict['z_source'] = row['z_source']
                    if 'theta_E' in row and pd.notna(row['theta_E']):
                        target_dict['theta_E'] = row['theta_E']
                    if 'e1_l' in row and pd.notna(row['e1_l']):
                        target_dict['e1_l'] = row['e1_l']
                    if 'e2_l' in row and pd.notna(row['e2_l']):
                        target_dict['e2_l'] = row['e2_l']
                    if 'grading' in row and pd.notna(row['grading']):
                        target_dict['grading'] = row['grading']
                    if 'type' in row and pd.notna(row['type']):
                        target_dict['type'] = row['type']
                    if 'ref' in row and pd.notna(row['ref']):
                        target_dict['ref'] = row['ref']
                    targets.append(target_dict)
                    
        if not targets:
            print(f"{RED}No hay objetivos para descargar. Especifica un catálogo (-c SLACS -d) o coordenadas (--ra ... --dec ...){ENDC}")
            return
            
        if args.limit > 0 and len(targets) > args.limit:
            print(f"{YELLOW}Limitando la descarga a los primeros {args.limit} objetos de {len(targets)}.{ENDC}")
            targets = targets[:args.limit]
        else:
            print(f"{CYAN}Descargando {len(targets)} objetos en total...{ENDC}")
            
        if args.ra is not None and args.dec is not None:
            catalog_folder = "CUSTOM"
        else:
            catalog_folder = args.catalog.upper()
            
        decam_dir = os.path.join(args.outdir, catalog_folder, 'DECam')
        hst_dir = os.path.join(args.outdir, catalog_folder, 'HST')
        comp_dir = os.path.join(args.outdir, catalog_folder, 'Comparaciones')
        
        print(f"\n{BOLD}Iniciando descargas para el catálogo '{catalog_folder}' con {args.workers} hilos en paralelo...{ENDC}\n")
        start_time = time.time()
        
        from concurrent.futures import ThreadPoolExecutor
        
        def process_item(item):
            i, t = item
            print(f"{BOLD}[{i}/{len(targets)}] Objetivo: {t['name']} (RA: {t['ra']}, DEC: {t['dec']}){ENDC}")
            
            decam_fits = os.path.join(decam_dir, f"decals_{t['name']}_ra{t['ra']:.4f}_dec{t['dec']:.4f}.fits")
            if args.source in ['decam', 'both']:
                download_decam_image(t['ra'], t['dec'], t['name'], decam_dir, pixscale=args.pixscale, size=args.size, layer=args.layer, fmt=args.fmt, fov_arcsec=args.fov)
                    
            hst_fits_local, space_tele = None, "HST"
            if args.source in ['hst', 'space', 'both']:
                hst_fits_local, space_tele = download_space_image(t['ra'], t['dec'], t['name'], hst_dir, radius=args.radius, filter_name=args.hst_filter)

            decam_jpg = os.path.join(decam_dir, f"decals_{t['name']}_ra{t['ra']:.4f}_dec{t['dec']:.4f}.jpg")
            decam_vir = os.path.join(decam_dir, f"decals_viridis_{t['name']}_ra{t['ra']:.4f}_dec{t['dec']:.4f}.png")
            hst_png = os.path.join(hst_dir, f"space_viridis_{t['name']}_ra{t['ra']:.4f}_dec{t['dec']:.4f}.png")
            if not os.path.exists(hst_png):
                hst_png = os.path.join(hst_dir, f"hst_viridis_{t['name']}_ra{t['ra']:.4f}_dec{t['dec']:.4f}.png")

            comp_path = os.path.join(comp_dir, f"comparison_{t['name']}_ra{t['ra']:.4f}_dec{t['dec']:.4f}.png")
            save_comparison_plot(decam_jpg, decam_vir, hst_png, comp_path, space_telescope=space_tele)

            if args.export_dataset:
                export_melo2025_dataset_tensor(
                    decam_fits_path=decam_fits,
                    hst_fits_path=hst_fits_local if hst_fits_local else "",
                    ra=t['ra'],
                    dec=t['dec'],
                    name=t['name'],
                    catalog_name=catalog_folder,
                    dataset_dir=args.dataset_dir,
                    dataset_fmt=args.dataset_fmt,
                    z_lens=t.get('z_lens', None),
                    z_source=t.get('z_source', None),
                    theta_E=t.get('theta_E', None),
                    e1_l=t.get('e1_l', None),
                    e2_l=t.get('e2_l', None),
                    grading=t.get('grading', None),
                    type_str=t.get('type', None),
                    ref=t.get('ref', None),
                    space_telescope=space_tele
                )

        indexed_targets = list(enumerate(targets, 1))
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            list(executor.map(process_item, indexed_targets))
            
        elapsed = time.time() - start_time
        print(f"\n{BOLD}{GREEN}======================================================{ENDC}")
        print(f"{BOLD}{GREEN}  ¡PROCESO COMPLETADO EN {elapsed:.1f} SEGUNDOS!{ENDC}")
        print(f"{GREEN}  - Catálogo: {catalog_folder}{ENDC}")
        print(f"{GREEN}  - Imágenes DECam guardadas en:       {os.path.abspath(decam_dir)}{ENDC}")
        print(f"{GREEN}  - Imágenes HST guardadas en:         {os.path.abspath(hst_dir)}{ENDC}")
        print(f"{GREEN}  - Gráficas comparativas guardadas en: {os.path.abspath(comp_dir)}{ENDC}")
        if args.export_dataset:
            print(f"{GREEN}  - Dataset Melo 2025 guardado en:      {os.path.abspath(args.dataset_dir)}{ENDC}")
        print(f"{BOLD}{GREEN}======================================================{ENDC}\n")

if __name__ == '__main__':
    main()
