import os
import sys
import glob
import subprocess
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QTabWidget, QLabel, QLineEdit, QSpinBox, QComboBox, QCheckBox,
    QPushButton, QTextEdit, QListWidget, QListWidgetItem, QSplitter,
    QGroupBox, QFormLayout, QFileDialog, QFrame, QStyle, QMessageBox,
    QDialog, QTableWidget, QTableWidgetItem, QHeaderView, QSizePolicy
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QFont, QPixmap, QIcon, QColor

# Path Detection
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if os.path.basename(SCRIPT_DIR) == 'training_dataset':
    DATASET_DIR = SCRIPT_DIR
    PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
else:
    DATASET_DIR = os.path.join(SCRIPT_DIR, 'training_dataset')
    PROJECT_ROOT = SCRIPT_DIR

sys.path.insert(0, DATASET_DIR)
sys.path.insert(0, PROJECT_ROOT)

# Worker Thread for Async Execution of Scripts
class ScriptRunnerThread(QThread):
    log_signal = pyqtSignal(str)
    finished_signal = pyqtSignal()

    def __init__(self, command, cwd=PROJECT_ROOT):
        super().__init__()
        self.command = command
        self.cwd = cwd

    def run(self):
        try:
            process = subprocess.Popen(
                self.command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                cwd=self.cwd
            )
            for line in process.stdout:
                self.log_signal.emit(line.rstrip())
            process.wait()
        except Exception as e:
            self.log_signal.emit(f"[ERROR] Error al ejecutar comando: {e}")
        finally:
            self.finished_signal.emit()

# Resizable Image Label that scales pixmap to fit widget bounds without window growth or clipping
class ResizableImageLabel(QLabel):
    def __init__(self, text=""):
        super().__init__(text)
        self.original_pixmap = None
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumSize(1, 1)

    def set_preview_pixmap(self, pixmap):
        self.original_pixmap = pixmap
        self.update_scaled_pixmap()

    def update_scaled_pixmap(self):
        if self.original_pixmap and not self.original_pixmap.isNull():
            w = max(10, self.width() - 4)
            h = max(10, self.height() - 4)
            scaled = self.original_pixmap.scaled(
                w, h,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )
            super().setPixmap(scaled)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.update_scaled_pixmap()

# Main Desktop GUI Window
class LenstronomyStudioApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Lenstronomy Dataset Studio - Lentes Gravitacionales (DECam / HST / Mocks)")
        self.resize(1300, 860)
        self.setStyleSheet(self.get_dark_stylesheet())

        self.runner_thread = None
        self.init_ui()
        self.refresh_stats()

    def get_dark_stylesheet(self):
        return """
            QMainWindow, QWidget {
                background-color: #0b0f19;
                color: #f1f5f9;
                font-family: 'Segoe UI', 'San Francisco', 'Helvetica Neue', sans-serif;
            }
            QTabWidget::pane {
                border: 1px solid rgba(255, 255, 255, 0.08);
                background-color: #121827;
                border-radius: 8px;
            }
            QTabBar::tab {
                background: rgba(255, 255, 255, 0.04);
                color: #94a3b8;
                padding: 10px 18px;
                margin-right: 4px;
                border-top-left-radius: 8px;
                border-top-right-radius: 8px;
                font-weight: 600;
                font-size: 13px;
            }
            QTabBar::tab:selected {
                background: #1e293b;
                color: #00f2fe;
                border-bottom: 2px solid #00f2fe;
            }
            QGroupBox {
                background: rgba(22, 28, 45, 0.7);
                border: 1px solid rgba(255, 255, 255, 0.08);
                border-radius: 10px;
                margin-top: 12px;
                padding-top: 14px;
                font-weight: bold;
                color: #00f2fe;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 14px;
                padding: 0 6px;
            }
            QLabel {
                color: #e2e8f0;
                font-size: 13px;
            }
            QLineEdit, QSpinBox, QComboBox {
                background: #060911;
                border: 1px solid rgba(255, 255, 255, 0.12);
                border-radius: 6px;
                padding: 6px 10px;
                color: #ffffff;
                font-size: 13px;
            }
            QLineEdit:focus, QSpinBox:focus, QComboBox:focus {
                border: 1px solid #00f2fe;
            }
            QPushButton {
                background-color: #0284c7;
                color: #ffffff;
                font-weight: bold;
                font-size: 13px;
                padding: 10px 16px;
                border-radius: 8px;
                border: none;
            }
            QPushButton:hover {
                background-color: #0369a1;
            }
            QPushButton:disabled {
                background-color: #334155;
                color: #64748b;
            }
            QTextEdit {
                background-color: #04060c;
                color: #38bdf8;
                font-family: 'Fira Code', 'Consolas', monospace;
                font-size: 12px;
                border: 1px solid rgba(0, 242, 254, 0.2);
                border-radius: 6px;
            }
            QListWidget {
                background-color: #060911;
                border: 1px solid rgba(255, 255, 255, 0.1);
                border-radius: 6px;
                color: #f1f5f9;
            }
            QListWidget::item:selected {
                background-color: #0284c7;
                color: #ffffff;
            }
        """

    def init_ui(self):
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(16, 16, 16, 16)
        main_layout.setSpacing(12)

        # Header Bar & Live Stats Badges & Exit Button
        header_layout = QHBoxLayout()

        title_layout = QVBoxLayout()
        title_label = QLabel("🌌 Lenstronomy Dataset Studio")
        title_label.setStyleSheet("font-size: 20px; font-weight: bold; color: #ffffff;")
        subtitle_label = QLabel("Generador Sintético, Descargador de Catálogos Reales y Visualizador de TFRecords")
        subtitle_label.setStyleSheet("font-size: 12px; color: #94a3b8;")
        title_layout.addWidget(title_label)
        title_layout.addWidget(subtitle_label)

        header_layout.addLayout(title_layout)
        header_layout.addStretch()

        # Stats Badges
        self.badge_tfrecords = QLabel("TFRecords: 0")
        self.badge_tfrecords.setStyleSheet("background: #1e293b; border: 1px solid rgba(255,255,255,0.1); padding: 6px 12px; border-radius: 6px; font-weight: bold; color: #38bdf8;")
        self.badge_tensors = QLabel("Tensors NPY: 0")
        self.badge_tensors.setStyleSheet("background: #1e293b; border: 1px solid rgba(255,255,255,0.1); padding: 6px 12px; border-radius: 6px; font-weight: bold; color: #a855f7;")
        self.badge_pngs = QLabel("PNGs Renderizados: 0")
        self.badge_pngs.setStyleSheet("background: #1e293b; border: 1px solid rgba(255,255,255,0.1); padding: 6px 12px; border-radius: 6px; font-weight: bold; color: #34d399;")

        header_layout.addWidget(self.badge_tfrecords)
        header_layout.addWidget(self.badge_tensors)
        header_layout.addWidget(self.badge_pngs)

        # EXIT BUTTON
        self.btn_exit = QPushButton("🚪 Salir")
        self.btn_exit.setStyleSheet("""
            QPushButton {
                background-color: #ef4444;
                color: #ffffff;
                font-weight: bold;
                font-size: 13px;
                padding: 8px 16px;
                border-radius: 6px;
                border: none;
            }
            QPushButton:hover {
                background-color: #dc2626;
            }
        """)
        self.btn_exit.setToolTip("Cerrar la aplicación Lenstronomy Dataset Studio")
        self.btn_exit.clicked.connect(self.close_application)
        header_layout.addWidget(self.btn_exit)

        main_layout.addLayout(header_layout)

        # Tab Widget
        self.tabs = QTabWidget()
        self.tabs.addTab(self.create_tab_mocks(), "🌌 Base Sintética (Mocks)")
        self.tabs.addTab(self.create_tab_real(), "🔭 Base Real (DECam/HST)")
        self.tabs.addTab(self.create_tab_real_viewer(), "🔬 Visualizador de Ejemplo Real")
        self.tabs.addTab(self.create_tab_converter(), "🖼️ Convertidor TFRecords -> PNG")
        self.tabs.addTab(self.create_tab_gallery(), "📊 Galería de Previsualización")

        main_layout.addWidget(self.tabs)

        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)

    def close_application(self):
        """Prompt confirmation before exiting GUI."""
        reply = QMessageBox.question(
            self, 'Confirmar Salida',
            '¿Estás seguro de que deseas salir de Lenstronomy Dataset Studio?',
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No
        )
        if reply == QMessageBox.StandardButton.Yes:
            QApplication.quit()

    # TAB 1: Synthetic Mocks
    def create_tab_mocks(self):
        widget = QWidget()
        layout = QHBoxLayout()

        left_box = QGroupBox("Parámetros de Simulación (Lenstronomy)")
        form_layout = QVBoxLayout()
        f_layout = QFormLayout()

        self.num_mocks_input = QSpinBox()
        self.num_mocks_input.setRange(1, 5000)
        self.num_mocks_input.setValue(10)
        f_layout.addRow("Número de Mocks a Generar:", self.num_mocks_input)

        self.mock_fmt_combo = QComboBox()
        self.mock_fmt_combo.addItems([
            "both (TFRecord .tfrecord y NumPy .npy)",
            "tfrecord (Solo TFRecord .tfrecord)",
            "npy (Solo NumPy .npy)"
        ])
        f_layout.addRow("Formato de Salida:", self.mock_fmt_combo)

        info_lbl = QLabel("Modelado Físico Melo et al. (2025):\n• Lente: Perfil SIE (θ_E ∈ [1.0, 2.2] arcsec)\n• Fuente: Sersic Elíptica con simulación de arcos\n• Rejilla 4 Canales: DECam (g, r, z) + HST (F814W)\n• FOV: 10\" x 10\" (200 x 200 píxeles, 0.05\"/px)")
        info_lbl.setStyleSheet("color: #94a3b8; font-size: 11px; padding: 8px; background: rgba(0,0,0,0.3); border-radius: 6px;")
        f_layout.addRow(info_lbl)

        form_layout.addLayout(f_layout)
        form_layout.addStretch()

        self.btn_run_mocks = QPushButton("🚀 Generar Base Sintética Mocks")
        self.btn_run_mocks.clicked.connect(self.run_mocks_generation)
        form_layout.addWidget(self.btn_run_mocks)

        left_box.setLayout(form_layout)

        right_box = QGroupBox("Terminal Log - Proceso Mocks")
        right_layout = QVBoxLayout()
        self.log_mocks = QTextEdit()
        self.log_mocks.setReadOnly(True)
        self.log_mocks.setPlaceholderText("Presiona 'Generar Base Sintética Mocks' para iniciar la simulación...")
        right_layout.addWidget(self.log_mocks)
        right_box.setLayout(right_layout)

        layout.addWidget(left_box, 1)
        layout.addWidget(right_box, 1)
        widget.setLayout(layout)
        return widget

    # TAB 2: Real Catalogs
    def create_tab_real(self):
        widget = QWidget()
        layout = QHBoxLayout()

        left_box = QGroupBox("Descarga de Catálogos Astronómicos Reales")
        form_layout = QVBoxLayout()

        cat_lbl = QLabel("Seleccionar Catálogos (Librería Lenscat / VizieR):")
        form_layout.addWidget(cat_lbl)

        cat_grid = QHBoxLayout()
        self.cat_checks = {}
        for cat in ['SLACS', 'BELLS', 'DES', 'SUGOHI', 'SQLS', 'CLASS', 'CONFIDENT', 'LENSCAT']:
            chk = QCheckBox(cat)
            if cat in ['SLACS', 'BELLS', 'DES', 'CONFIDENT']:
                chk.setChecked(True)
            self.cat_checks[cat] = chk
            cat_grid.addWidget(chk)
        form_layout.addLayout(cat_grid)

        f_layout = QFormLayout()

        self.real_limit_input = QSpinBox()
        self.real_limit_input.setRange(0, 1000)
        self.real_limit_input.setValue(10)
        f_layout.addRow("Límite por Catálogo (0 para Todo):", self.real_limit_input)

        self.real_source_combo = QComboBox()
        self.real_source_combo.addItems([
            "both (DECam / DECaLS + HST MAST)",
            "decam (Solo DECam / DECaLS óptico g, r, z)",
            "hst (Solo HST espacial F814W)"
        ])
        f_layout.addRow("Origen de Imágenes Cutout:", self.real_source_combo)

        self.real_fmt_combo = QComboBox()
        self.real_fmt_combo.addItems([
            "both (TFRecord .tfrecord y NumPy .npy)",
            "tfrecord (Solo TFRecord .tfrecord)",
            "npy (Solo NumPy .npy)"
        ])
        f_layout.addRow("Formato de Salida del Dataset:", self.real_fmt_combo)

        form_layout.addLayout(f_layout)
        form_layout.addStretch()

        self.btn_run_real = QPushButton("📥 Descargar y Procesar Catálogos Reales")
        self.btn_run_real.clicked.connect(self.run_real_download)
        form_layout.addWidget(self.btn_run_real)

        left_box.setLayout(form_layout)

        right_box = QGroupBox("Terminal Log - Descarga Real")
        right_layout = QVBoxLayout()
        self.log_real = QTextEdit()
        self.log_real.setReadOnly(True)
        self.log_real.setPlaceholderText("Presiona 'Descargar y Procesar Catálogos Reales' para iniciar la consulta...")
        right_layout.addWidget(self.log_real)
        right_box.setLayout(right_layout)

        layout.addWidget(left_box, 1)
        layout.addWidget(right_box, 1)
        widget.setLayout(layout)
        return widget

    # TAB 3: Visualizador de Ejemplo Real (NEW FEATURE)
    def create_tab_real_viewer(self):
        widget = QWidget()
        layout = QVBoxLayout()

        # Top Control Bar
        top_bar = QHBoxLayout()

        lbl_filter = QLabel("Catálogo:")
        lbl_filter.setStyleSheet("font-weight: bold; color: #00f2fe;")
        top_bar.addWidget(lbl_filter)

        self.real_cat_filter = QComboBox()
        self.real_cat_filter.addItems(["Todos los Catálogos Reales", "CONFIDENT", "LENSCAT", "DES", "SLACS", "BELLS", "SUGOHI", "SQLS", "CLASS"])
        self.real_cat_filter.currentIndexChanged.connect(self.populate_real_lenses_dropdown)
        top_bar.addWidget(self.real_cat_filter)

        lbl_lens = QLabel("Seleccionar Lente Real:")
        lbl_lens.setStyleSheet("font-weight: bold; color: #ffffff;")
        top_bar.addWidget(lbl_lens)

        self.real_lens_combo = QComboBox()
        self.real_lens_combo.setMinimumWidth(360)
        top_bar.addWidget(self.real_lens_combo)

        btn_load_real = QPushButton("👁️ Cargar y Visualizar Lente Real")
        btn_load_real.setStyleSheet("background-color: #7f00ff; color: #ffffff; font-weight: bold;")
        btn_load_real.clicked.connect(self.load_selected_real_lens)
        top_bar.addWidget(btn_load_real)

        btn_show_all_params = QPushButton("📋 Tabla de Parámetros Físicos")
        btn_show_all_params.setStyleSheet("background-color: #00f2fe; color: #060911; font-weight: bold;")
        btn_show_all_params.clicked.connect(self.show_physical_parameters_dialog)
        top_bar.addWidget(btn_show_all_params)

        top_bar.addStretch()
        layout.addLayout(top_bar)

        # Splitter Layout: Left Details Card & Right 4-Panel Render Image
        splitter = QSplitter(Qt.Orientation.Horizontal)

        # Left Metadata Details GroupBox
        details_box = QGroupBox("Detalles y Metadatos del Objeto Real")
        details_layout = QVBoxLayout()

        self.real_details_text = QTextEdit()
        self.real_details_text.setReadOnly(True)
        self.real_details_text.setStyleSheet("background-color: #060911; color: #f1f5f9; font-size: 13px; border: 1px solid rgba(255,255,255,0.1);")
        self.real_details_text.setHtml("<p style='color: #94a3b8;'>Selecciona un lente real del menú desplegable superior y haz clic en 'Cargar y Visualizar Lente Real' para examinar sus atributos físicos e imágenes de DECam / HST.</p>")
        details_layout.addWidget(self.real_details_text)
        details_box.setLayout(details_layout)

        splitter.addWidget(details_box)

        # Right Preview Image Box
        preview_box = QGroupBox("Panel de Visualización Melo et al. (2025)")
        preview_layout = QVBoxLayout()

        self.real_preview_label = ResizableImageLabel("Sin imagen seleccionada")
        self.real_preview_label.setStyleSheet("background: #060911; border: 1px solid rgba(255,255,255,0.1); border-radius: 8px; color: #94a3b8;")
        preview_layout.addWidget(self.real_preview_label)
        preview_box.setLayout(preview_layout)

        splitter.addWidget(preview_box)
        splitter.setSizes([400, 880])

        layout.addWidget(splitter)
        widget.setLayout(layout)

        # Initial Population of Dropdown
        self.populate_real_lenses_dropdown()
        return widget

    def populate_real_lenses_dropdown(self):
        self.real_lens_combo.clear()
        selected_cat = self.real_cat_filter.currentText()

        tfrecords_dir = os.path.join(DATASET_DIR, 'tfrecords')
        if not os.path.exists(tfrecords_dir):
            return

        all_tfrecords = sorted(glob.glob(os.path.join(tfrecords_dir, "**", "*.tfrecord"), recursive=True))
        real_tfrecords = [f for f in all_tfrecords if not os.path.basename(f).startswith("mock_lens_")]

        if selected_cat != "Todos los Catálogos Reales":
            real_tfrecords = [f for f in real_tfrecords if selected_cat.upper() in f.upper()]

        for f in real_tfrecords:
            basename = os.path.splitext(os.path.basename(f))[0]
            parent_cat = os.path.basename(os.path.dirname(f))
            display_name = f"[{parent_cat}] {basename}" if parent_cat and parent_cat.lower() != 'tfrecords' else basename
            self.real_lens_combo.addItem(display_name, f)

        if self.real_lens_combo.count() == 0:
            self.real_lens_combo.addItem("No se encontraron TFRecords reales descargados", None)

    def load_selected_real_lens(self):
        tfrecord_path = self.real_lens_combo.currentData()
        if not tfrecord_path or not os.path.exists(tfrecord_path):
            self.real_details_text.setHtml("<p style='color: #ef4444;'>No hay un archivo TFRecord real seleccionado. Utiliza la pestaña 'Base Real' para descargar recortes de catálogos astronómicos.</p>")
            return

        basename = os.path.splitext(os.path.basename(tfrecord_path))[0]
        parent_cat = os.path.basename(os.path.dirname(tfrecord_path))
        
        png_path = os.path.join(DATASET_DIR, "tfrecords_pngs", parent_cat, f"{basename}.png")
        if not os.path.exists(png_path):
            matches = glob.glob(os.path.join(DATASET_DIR, "tfrecords_pngs", "**", f"{basename}.png"), recursive=True)
            if matches:
                png_path = matches[0]
            else:
                png_path = os.path.join(DATASET_DIR, "tfrecords_pngs", f"{basename}.png")

        # If PNG preview doesn't exist yet, generate it on the fly!
        if not os.path.exists(png_path):
            self.real_details_text.setHtml(f"<p style='color: #f59e0b;'>Generando vista previa PNG para {basename}...</p>")
            script_path = os.path.join(DATASET_DIR, 'visualize_tfrecords.py')
            cmd = [sys.executable, script_path, '-p', tfrecord_path, '--outdir', os.path.join(DATASET_DIR, "tfrecords_pngs")]
            subprocess.run(cmd, cwd=PROJECT_ROOT)
            matches = glob.glob(os.path.join(DATASET_DIR, "tfrecords_pngs", "**", f"{basename}.png"), recursive=True)
            if matches:
                png_path = matches[0]

        # Display Image Preview
        if os.path.exists(png_path):
            pixmap = QPixmap(png_path)
            self.real_preview_label.set_preview_pixmap(pixmap)

        # Parse Metadata from TFRecord for display card
        try:
            import tensorflow as tf
            try:
                from training_dataset.visualize_tfrecords import parse_tfrecord_with_metadata
            except ImportError:
                from visualize_tfrecords import parse_tfrecord_with_metadata

            raw_ds = tf.data.TFRecordDataset([tfrecord_path])
            for proto in raw_ds.take(1):
                _, decals_3ch, hst_1ch, meta = parse_tfrecord_with_metadata(proto)

                html_info = f"""
                <h3 style='color: #00f2fe; margin-bottom: 6px;'>🌌 Objeto Real: {meta['name']}</h3>
                <p><b>Catálogo Origen:</b> <span style='color: #a855f7;'>{meta['catalog'] if meta['catalog'] else 'Astronomical Catalog'}</span></p>
                <hr style='border-color: rgba(255,255,255,0.1); margin: 8px 0;'>
                <p><b>• Coordenadas Equinocciales:</b><br>RA: <span style='color: #38bdf8;'>{meta['ra']:.4f}°</span> | DEC: <span style='color: #38bdf8;'>{meta['dec']:.4f}°</span></p>
                """
                if meta.get('grading'):
                    html_info += f"<p><b>• Clasificación (Grading):</b> <span style='color: #10b981; font-weight: bold;'>{meta['grading']}</span></p>"
                if meta.get('type'):
                    html_info += f"<p><b>• Tipo de Sistema:</b> {meta['type']}</p>"
                if meta.get('ref'):
                    html_info += f"<p><b>• Referencia Bibliográfica:</b> <span style='color: #cbd5e1; font-size: 11px;'>{meta['ref']}</span></p>"

                if meta['z_lens'] is not None:
                    html_info += f"<p><b>• Redshift Lente (z<sub>l</sub>):</b> {meta['z_lens']:.3f}</p>"
                if meta['z_source'] is not None:
                    html_info += f"<p><b>• Redshift Fuente (z<sub>s</sub>):</b> {meta['z_source']:.3f}</p>"

                html_info += "<hr style='border-color: rgba(255,255,255,0.1); margin: 8px 0;'><h4 style='color: #f59e0b;'>🔬 Característica Físicas:</h4>"
                if meta['theta_E'] is not None:
                    html_info += f"<p><b>• Radio de Einstein (&theta;<sub>E</sub>):</b> {meta['theta_E']:.2f} arcsec</p>"
                else:
                    html_info += "<p><b>• Radio de Einstein (&theta;<sub>E</sub>):</b> Sistema Confirmado (Observacional)</p>"

                if meta['e1_l'] is not None and meta['e2_l'] is not None:
                    import numpy as np
                    e_mag = np.sqrt(meta['e1_l']**2 + meta['e2_l']**2)
                    html_info += f"<p><b>• Elipticidad (e<sub>1</sub>, e<sub>2</sub>):</b> ({meta['e1_l']:.2f}, {meta['e2_l']:.2f})</p>"
                    html_info += f"<p><b>• Magnitud Elipticidad (|e|):</b> {e_mag:.2f}</p>"

                html_info += """
                <hr style='border-color: rgba(255,255,255,0.1); margin: 8px 0;'>
                <h4 style='color: #34d399;'>📡 Cobertura de Telescopios & Especificaciones:</h4>
                """
                if decals_3ch is not None:
                    html_info += "<p><b>✓ DECam / DECaLS:</b> Cobertura Terrestre (Filtros g, r, z)</p>"
                else:
                    html_info += "<p style='color: #ef4444;'><b>✗ DECam / DECaLS:</b> Sin cobertura de recorte</p>"

                space_tele_name = meta.get('space_telescope', 'JWST/HST')
                if not space_tele_name or space_tele_name == 'NONE':
                    space_tele_name = 'JWST/HST'

                if hst_1ch is not None:
                    html_info += f"<p><b>✓ Observatorio Espacial ({space_tele_name}):</b> Cobertura de Alta Resolución (MAST)</p>"
                else:
                    html_info += f"<p style='color: #f59e0b;'><b>! MAST Espacial ({space_tele_name}):</b> Sin observaciones en MAST</p>"

                html_info += f"""
                <p><b>• Campo de Visión (FOV):</b> 10" x 10" (200 x 200 px, 0.05"/px)</p>
                <hr style='border-color: rgba(255,255,255,0.1); margin: 8px 0;'>
                <p style='font-size: 11px; color: #94a3b8;'><b>Ruta TFRecord:</b><br>{tfrecord_path}</p>
                """
                self.real_details_text.setHtml(html_info)
        except Exception as err:
            self.real_details_text.setHtml(f"<p style='color: #ef4444;'>Error al leer metadatos del TFRecord: {err}</p>")

    def show_physical_parameters_dialog(self):
        try:
            from print_parameters import load_all_parameters
        except ImportError:
            from training_dataset.print_parameters import load_all_parameters

        import pandas as pd
        dialog = QDialog(self)
        dialog.setWindowTitle("📋 Tabla de Parámetros Físicos de Todos los Sistemas")
        dialog.resize(1100, 600)
        dialog.setStyleSheet("background-color: #0d1117; color: #f1f5f9;")

        d_layout = QVBoxLayout()

        # Header controls
        h_layout = QHBoxLayout()
        lbl = QLabel("Filtrar por Catálogo:")
        lbl.setStyleSheet("font-weight: bold; color: #00f2fe;")
        combo_cat = QComboBox()
        combo_cat.addItems(["Todos", "SLACS", "DES", "SUGOHI", "SQLS", "BELLS", "CLASS", "MOCK"])

        btn_export_csv = QPushButton("💾 Exportar a CSV")
        btn_export_csv.setStyleSheet("background-color: #10b981; color: #ffffff; font-weight: bold;")

        btn_print_console = QPushButton("🖨️ Imprimir en Terminal")
        btn_print_console.setStyleSheet("background-color: #3b82f6; color: #ffffff; font-weight: bold;")

        h_layout.addWidget(lbl)
        h_layout.addWidget(combo_cat)
        h_layout.addStretch()
        h_layout.addWidget(btn_export_csv)
        h_layout.addWidget(btn_print_console)
        d_layout.addLayout(h_layout)

        # Table Widget
        table = QTableWidget()
        table.setColumnCount(10)
        table.setHorizontalHeaderLabels([
            "Nombre", "Catálogo", "RA (deg)", "DEC (deg)",
            "z_lens", "z_source", "θ_E (arcsec)", "q (b/a)", "e1", "e2"
        ])
        table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        table.setStyleSheet("""
            QTableWidget {
                background-color: #060911;
                gridline-color: rgba(255,255,255,0.1);
                font-size: 12px;
            }
            QHeaderView::section {
                background-color: #1e293b;
                color: #38bdf8;
                font-weight: bold;
                padding: 4px;
                border: 1px solid rgba(255,255,255,0.1);
            }
        """)

        df_all = load_all_parameters()

        def populate_table():
            table.setRowCount(0)
            cat_sel = combo_cat.currentText()
            df_filtered = df_all if cat_sel == "Todos" else df_all[df_all['catalog'].str.upper() == cat_sel.upper()]

            table.setRowCount(len(df_filtered))
            for i, (_, r) in enumerate(df_filtered.iterrows()):
                table.setItem(i, 0, QTableWidgetItem(str(r['name'])))
                table.setItem(i, 1, QTableWidgetItem(str(r['catalog'])))
                table.setItem(i, 2, QTableWidgetItem(f"{r['ra']:.4f}" if pd.notna(r['ra']) else "-"))
                table.setItem(i, 3, QTableWidgetItem(f"{r['dec']:.4f}" if pd.notna(r['dec']) else "-"))
                table.setItem(i, 4, QTableWidgetItem(f"{r['z_lens']:.3f}" if pd.notna(r['z_lens']) else "-"))
                table.setItem(i, 5, QTableWidgetItem(f"{r['z_source']:.3f}" if pd.notna(r['z_source']) else "-"))
                table.setItem(i, 6, QTableWidgetItem(f"{r['theta_E']:.2f}" if pd.notna(r['theta_E']) else "-"))
                table.setItem(i, 7, QTableWidgetItem(f"{r['q_ratio']:.2f}" if pd.notna(r['q_ratio']) else "-"))
                table.setItem(i, 8, QTableWidgetItem(f"{r['e1_l']:.2f}" if pd.notna(r['e1_l']) else "-"))
                table.setItem(i, 9, QTableWidgetItem(f"{r['e2_l']:.2f}" if pd.notna(r['e2_l']) else "-"))

        combo_cat.currentIndexChanged.connect(populate_table)
        populate_table()

        def do_export_csv():
            path, _ = QFileDialog.getSaveFileName(dialog, "Guardar Parámetros como CSV", os.path.join(DATASET_DIR, "sistemas_parametros_fisicos.csv"), "CSV Files (*.csv)")
            if path:
                df_all.to_csv(path, index=False)
                QMessageBox.information(dialog, "Éxito", f"Tabla de parámetros exportada en:\n{path}")

        def do_print_console():
            try:
                from print_parameters import print_summary_table
            except ImportError:
                from training_dataset.print_parameters import print_summary_table
            print_summary_table(df_all, catalog_filter=None if combo_cat.currentText()=="Todos" else combo_cat.currentText())
            QMessageBox.information(dialog, "Terminal", "La tabla de parámetros ha sido impresa en la consola/terminal.")

        btn_export_csv.clicked.connect(do_export_csv)
        btn_print_console.clicked.connect(do_print_console)

        d_layout.addWidget(table)

        btn_close = QPushButton("Cerrar")
        btn_close.clicked.connect(dialog.accept)
        d_layout.addWidget(btn_close)

        dialog.setLayout(d_layout)
        dialog.exec()

    # TAB 4: Converter
    def create_tab_converter(self):
        widget = QWidget()
        layout = QHBoxLayout()

        left_box = QGroupBox("Convertir y Renderizar TFRecords a PNG")
        form_layout = QVBoxLayout()
        f_layout = QFormLayout()

        self.convert_pattern_input = QLineEdit()
        self.convert_pattern_input.setPlaceholderText("Ej. DES, MOCK, SLACS (vacío para TODOS)")
        f_layout.addRow("Filtrar por Nombre/Catálogo:", self.convert_pattern_input)

        self.convert_outdir_input = QLineEdit(os.path.join(DATASET_DIR, "tfrecords_pngs"))
        f_layout.addRow("Directorio de PNGs:", self.convert_outdir_input)

        form_layout.addLayout(f_layout)
        form_layout.addStretch()

        self.btn_run_converter = QPushButton("🎨 Renderizar TFRecords a PNG")
        self.btn_run_converter.clicked.connect(self.run_tfrecord_converter)
        form_layout.addWidget(self.btn_run_converter)

        left_box.setLayout(form_layout)

        right_box = QGroupBox("Terminal Log - Renderizado PNG")
        right_layout = QVBoxLayout()
        self.log_converter = QTextEdit()
        self.log_converter.setReadOnly(True)
        self.log_converter.setPlaceholderText("Presiona 'Renderizar TFRecords a PNG' para iniciar...")
        right_layout.addWidget(self.log_converter)
        right_box.setLayout(right_layout)

        layout.addWidget(left_box, 1)
        layout.addWidget(right_box, 1)
        widget.setLayout(layout)
        return widget

    # TAB 5: Gallery Preview
    def create_tab_gallery(self):
        widget = QWidget()
        layout = QVBoxLayout()

        header = QHBoxLayout()
        lbl = QLabel("Explorador de Previsualizaciones PNG (4 Paneles Melo et al. 2025)")
        lbl.setStyleSheet("font-size: 14px; font-weight: bold; color: #00f2fe;")
        btn_refresh = QPushButton("🔄 Actualizar Galería")
        btn_refresh.clicked.connect(self.load_gallery)

        header.addWidget(lbl)
        header.addStretch()
        header.addWidget(btn_refresh)
        layout.addLayout(header)

        splitter = QSplitter(Qt.Orientation.Horizontal)

        self.gallery_list = QListWidget()
        self.gallery_list.itemSelectionChanged.connect(self.display_selected_image)
        splitter.addWidget(self.gallery_list)

        self.image_preview_label = ResizableImageLabel("Selecciona un archivo PNG de la lista para previsualizar")
        self.image_preview_label.setStyleSheet("background: #060911; border: 1px solid rgba(255,255,255,0.1); border-radius: 8px; color: #94a3b8;")
        splitter.addWidget(self.image_preview_label)

        splitter.setSizes([320, 960])
        layout.addWidget(splitter)

        widget.setLayout(layout)
        return widget

    # Actions & Thread Handlers
    def run_mocks_generation(self):
        num_mocks = self.num_mocks_input.value()
        fmt_str = self.mock_fmt_combo.currentText().split(' ')[0]

        script_path = os.path.join(DATASET_DIR, 'import_images_DECam.py')
        cmd = [
            sys.executable, script_path,
            '-m',
            '--num-mocks', str(num_mocks),
            '--dataset-fmt', fmt_str,
            '--dataset-dir', DATASET_DIR
        ]

        self.log_mocks.clear()
        self.log_mocks.append("🚀 Iniciando simulación de Mocks con Lenstronomy...\n")
        self.btn_run_mocks.setEnabled(False)

        self.runner_thread = ScriptRunnerThread(cmd)
        self.runner_thread.log_signal.connect(self.log_mocks.append)
        self.runner_thread.finished_signal.connect(lambda: self.on_task_finished(self.btn_run_mocks))
        self.runner_thread.start()

    def run_real_download(self):
        selected_cats = [cat for cat, chk in self.cat_checks.items() if chk.isChecked()]
        if not selected_cats:
            self.log_real.setText("⚠️ Error: Selecciona al menos un catálogo.")
            return

        limit = self.real_limit_input.value()
        source_str = self.real_source_combo.currentText().split(' ')[0]
        fmt_str = self.real_fmt_combo.currentText().split(' ')[0]

        script_path = os.path.join(DATASET_DIR, 'import_images_DECam.py')
        cmd = [
            sys.executable, script_path,
            '-c', selected_cats[0],
            '-s', '-d',
            '--limit', str(limit),
            '--source', source_str,
            '--export-dataset',
            '--dataset-fmt', fmt_str,
            '--dataset-dir', DATASET_DIR
        ]

        self.log_real.clear()
        self.log_real.append(f"📡 Consultando VizieR/MAST para catálogos: {', '.join(selected_cats)}...\n")
        self.btn_run_real.setEnabled(False)

        self.runner_thread = ScriptRunnerThread(cmd)
        self.runner_thread.log_signal.connect(self.log_real.append)
        self.runner_thread.finished_signal.connect(lambda: self.on_task_finished(self.btn_run_real))
        self.runner_thread.start()

    def run_tfrecord_converter(self):
        pattern = self.convert_pattern_input.text().strip()
        outdir = self.convert_outdir_input.text().strip()

        script_path = os.path.join(DATASET_DIR, 'visualize_tfrecords.py')
        cmd = [
            sys.executable, script_path,
            '-p', os.path.join(DATASET_DIR, 'tfrecords'),
            '-a',
            '--outdir', outdir
        ]
        if pattern:
            cmd.extend(['-n', pattern])

        self.log_converter.clear()
        self.log_converter.append("🎨 Iniciando renderizado de TFRecords a PNG...\n")
        self.btn_run_converter.setEnabled(False)

        self.runner_thread = ScriptRunnerThread(cmd)
        self.runner_thread.log_signal.connect(self.log_converter.append)
        self.runner_thread.finished_signal.connect(lambda: self.on_task_finished(self.btn_run_converter))
        self.runner_thread.start()

    def on_task_finished(self, button):
        button.setEnabled(True)
        self.refresh_stats()
        self.load_gallery()
        self.populate_real_lenses_dropdown()

    def refresh_stats(self):
        tfrecords_dir = os.path.join(DATASET_DIR, 'tfrecords')
        tensors_dir = os.path.join(DATASET_DIR, 'tensors')
        pngs_dir = os.path.join(DATASET_DIR, 'tfrecords_pngs')

        tfrecords_count = len(glob.glob(os.path.join(tfrecords_dir, '**', '*.tfrecord'), recursive=True)) if os.path.exists(tfrecords_dir) else 0
        tensors_count = len(glob.glob(os.path.join(tensors_dir, '**', '*.npy'), recursive=True)) if os.path.exists(tensors_dir) else 0
        pngs_count = len(glob.glob(os.path.join(pngs_dir, '**', '*.png'), recursive=True)) if os.path.exists(pngs_dir) else 0

        self.badge_tfrecords.setText(f"TFRecords: {tfrecords_count}")
        self.badge_tensors.setText(f"Tensors NPY: {tensors_count}")
        self.badge_pngs.setText(f"PNGs Renderizados: {pngs_count}")

    def load_gallery(self):
        pngs_dir = os.path.join(DATASET_DIR, 'tfrecords_pngs')
        self.gallery_list.clear()

        if not os.path.exists(pngs_dir):
            return

        png_paths = sorted(glob.glob(os.path.join(pngs_dir, '**', '*.png'), recursive=True))
        for p in png_paths:
            rel_path = os.path.relpath(p, pngs_dir)
            item = QListWidgetItem(rel_path)
            item.setData(Qt.ItemDataRole.UserRole, p)
            self.gallery_list.addItem(item)

    def display_selected_image(self):
        items = self.gallery_list.selectedItems()
        if not items:
            return

        img_path = items[0].data(Qt.ItemDataRole.UserRole)
        if os.path.exists(img_path):
            pixmap = QPixmap(img_path)
            self.image_preview_label.set_preview_pixmap(pixmap)

def main():
    app = QApplication(sys.argv)
    window = LenstronomyStudioApp()
    window.show()
    sys.exit(app.exec())

if __name__ == '__main__':
    main()
