"""
Main application window for VapourSynth Image Upscaler.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import TYPE_CHECKING
from urllib.parse import urlparse
from urllib.request import urlretrieve

from PySide6.QtWidgets import (
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QGridLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QFileDialog,
    QCheckBox,
    QProgressBar,
    QGroupBox,
    QMessageBox,
    QComboBox,
    QDialog,
)
from PySide6.QtGui import QPixmap, QGuiApplication, QDragEnterEvent, QDropEvent, QImage
from PySide6.QtCore import Qt, QSize, QEvent, QTimer, Slot

from ..core.constants import (
    GUI_INPUT_TMP_ROOT,
    SUPPORTED_IMAGE_EXTENSIONS,
    VALID_MODEL_SCALES,
    DEFAULT_OUTPUT_PATH,
    DEFAULT_ONNX_PATH,
)
from ..core.config import Config
from ..core.utils import cleanup_gui_input_tmp, format_time_hms
from .dialogs import CustomResolutionDialog
from .worker_thread import UpscaleWorkerThread

if TYPE_CHECKING:
    pass


class MainWindow(QMainWindow):
    """
    Main application window.

    Provides the UI for:
    - Selecting input files/folders
    - Configuring output options
    - Selecting ONNX model and processing parameters
    - Starting and monitoring the upscale process
    """

    def __init__(self):
        super().__init__()
        self.setWindowTitle("VapourSynth Upscaler (vsmlrt SR)")

        # State
        self._input_items: list[Path] = []
        self._output_path: Path | None = None
        self._current_input_width: int = 0
        self._current_input_height: int = 0

        # Timing / ETA state
        self._elapsed_timer: QTimer | None = None
        self._batch_start_time: float = 0.0
        self._total_files_in_batch: int = 0
        self._completed_files_in_batch: int = 0
        self._current_avg_per_image: float = 0.0

        # Custom resolution / secondary state
        self._custom_res_enabled = False
        self._custom_res_width = 0
        self._custom_res_height = 0
        self._custom_res_maintain_ar = True
        self._secondary_enabled = False
        self._secondary_mode = "width"
        self._secondary_width = 1920
        self._secondary_height = 1080

        # Worker thread reference
        self._worker_thread: UpscaleWorkerThread | None = None

        # Create widgets
        self._create_widgets()
        self._build_ui()
        self._connect_signals()

        # Load saved settings
        self._load_settings()

        # Enable drag & drop
        self.setAcceptDrops(True)

        # Intercept Ctrl+V on the input field
        self._input_edit.installEventFilter(self)

    def _create_widgets(self) -> None:
        """Create all UI widgets."""
        # Input/output
        self._input_edit = QLineEdit()
        self._output_edit = QLineEdit()
        self._onnx_edit = QLineEdit()
        self._onnx_edit.setText(DEFAULT_ONNX_PATH)

        # Tile and scale combos
        self._tile_w_combo = QComboBox()
        self._tile_h_combo = QComboBox()
        self._model_scale_combo = QComboBox()

        for combo in (self._tile_w_combo, self._tile_h_combo):
            combo.setEditable(True)
            combo.addItems(["512", "768", "1024", "1088", "1536", "1920"])

        self._tile_w_combo.setCurrentText("1088")
        self._tile_h_combo.setCurrentText("1920")

        self._model_scale_combo.setEditable(False)
        self._model_scale_combo.addItems([str(s) for s in VALID_MODEL_SCALES])
        self._model_scale_combo.setCurrentText("4")

        # Checkboxes
        self._same_dir_check = QCheckBox("Save next to input with suffix:")
        self._same_dir_suffix_edit = QLineEdit("_upscaled")
        self._append_model_suffix_check = QCheckBox("Append model suffix")
        self._overwrite_check = QCheckBox("Overwrite")
        self._overwrite_check.setChecked(True)
        self._alpha_check = QCheckBox("Use alpha for transparent formats (PNG/GIF/WEBP)")

        # Precision checkboxes
        self._fp16_check = QCheckBox("fp16")
        self._bf16_check = QCheckBox("bf16")
        self._bf16_check.setChecked(True)
        self._tf32_check = QCheckBox("tf32")

        # Labels
        self._current_file_label = QLabel("Current file: (none)")
        self._progress_label = QLabel("Idle")
        self._time_label = QLabel("Elapsed: 00:00:00 | ETA: --:--:--")
        self._avg_label = QLabel("Avg per image: -")
        self._progress_bar = QProgressBar()
        self._progress_bar.setRange(0, 100)
        self._thumbnail_label = QLabel("(no thumbnail)")
        self._thumbnail_label.setAlignment(Qt.AlignCenter)
        self._thumbnail_label.setMinimumSize(QSize(200, 200))

        # Buttons
        self._btn_in_file = QPushButton("Browse File")
        self._btn_in_folder = QPushButton("Browse Folder")
        self._btn_out = QPushButton("Browse")
        self._btn_clear_output = QPushButton("Clear")
        self._btn_onnx = QPushButton("Browse ONNX")
        self._start_button = QPushButton("Start")
        self._cancel_button = QPushButton("Cancel")
        self._cancel_button.setEnabled(False)
        self._custom_res_button = QPushButton("Custom resolution / secondary...")

    def _build_ui(self) -> None:
        """Build the main UI layout."""
        central = QWidget()
        self.setCentralWidget(central)

        main_layout = QGridLayout()
        central.setLayout(main_layout)

        row = 0

        # Input row
        main_layout.addWidget(QLabel("Input file/folder(s):"), row, 0)
        main_layout.addWidget(self._input_edit, row, 1)
        main_layout.addWidget(self._btn_in_file, row, 2)
        main_layout.addWidget(self._btn_in_folder, row, 3)
        row += 1

        # Output row
        main_layout.addWidget(QLabel("Output folder:"), row, 0)
        main_layout.addWidget(self._output_edit, row, 1)
        main_layout.addWidget(self._btn_out, row, 2)
        main_layout.addWidget(self._btn_clear_output, row, 3)
        row += 1

        # Hint
        hint = QLabel(
            "Tip: You can drag & drop multiple images/folders/URLs, "
            "or paste an image from clipboard into the input box (Ctrl+V)."
        )
        main_layout.addWidget(hint, row, 0, 1, 4)
        row += 1

        # ONNX row
        main_layout.addWidget(QLabel("ONNX Model:"), row, 0)
        main_layout.addWidget(self._onnx_edit, row, 1)
        main_layout.addWidget(self._btn_onnx, row, 2)
        row += 1

        # Tile / model scale group
        tile_box = QGroupBox("SR tile / shape / scale")
        tile_layout = QHBoxLayout()
        tile_box.setLayout(tile_layout)
        tile_layout.addWidget(QLabel("Tile / Shape W:"))
        tile_layout.addWidget(self._tile_w_combo)
        tile_layout.addWidget(QLabel("Tile / Shape H:"))
        tile_layout.addWidget(self._tile_h_combo)
        tile_layout.addWidget(QLabel("Model scale (SR):"))
        tile_layout.addWidget(self._model_scale_combo)
        main_layout.addWidget(tile_box, row, 0, 1, 4)
        row += 1

        # Precision options
        prec_box = QGroupBox("vsmlrt Backend.TRT options")
        prec_layout = QHBoxLayout()
        prec_box.setLayout(prec_layout)
        prec_layout.addWidget(self._fp16_check)
        prec_layout.addWidget(self._bf16_check)
        prec_layout.addWidget(self._tf32_check)
        prec_layout.addStretch()
        main_layout.addWidget(prec_box, row, 0, 1, 4)
        row += 1

        # Same-dir + suffix
        same_dir_layout = QHBoxLayout()
        same_dir_container = QWidget()
        same_dir_container.setLayout(same_dir_layout)
        same_dir_layout.addWidget(self._same_dir_check)
        same_dir_layout.addWidget(self._same_dir_suffix_edit)
        same_dir_layout.addWidget(QLabel("(empty = same name)"))
        same_dir_layout.addStretch()
        main_layout.addWidget(same_dir_container, row, 0, 1, 4)
        row += 1

        # Options row
        opts_layout = QHBoxLayout()
        opts_container = QWidget()
        opts_container.setLayout(opts_layout)
        opts_layout.addWidget(self._overwrite_check)
        opts_layout.addWidget(self._alpha_check)
        opts_layout.addWidget(self._append_model_suffix_check)
        opts_layout.addWidget(self._custom_res_button)
        opts_layout.addStretch()
        main_layout.addWidget(opts_container, row, 0, 1, 4)
        row += 1

        # Info labels
        main_layout.addWidget(self._current_file_label, row, 0, 1, 4)
        row += 1
        main_layout.addWidget(self._progress_label, row, 0, 1, 4)
        row += 1
        main_layout.addWidget(self._progress_bar, row, 0, 1, 4)
        row += 1
        main_layout.addWidget(self._time_label, row, 0, 1, 4)
        row += 1
        main_layout.addWidget(self._avg_label, row, 0, 1, 4)
        row += 1

        # Thumbnail
        thumb_box = QGroupBox("Input thumbnail")
        thumb_layout = QVBoxLayout()
        thumb_box.setLayout(thumb_layout)
        thumb_layout.addWidget(self._thumbnail_label)
        main_layout.addWidget(thumb_box, 0, 4, row, 1)

        # Buttons at bottom
        btn_layout = QHBoxLayout()
        btn_container = QWidget()
        btn_container.setLayout(btn_layout)
        btn_layout.addWidget(self._start_button)
        btn_layout.addWidget(self._cancel_button)
        main_layout.addWidget(btn_container, row, 0, 1, 5)

        self.resize(1150, 680)

    def _connect_signals(self) -> None:
        """Connect widget signals to slots."""
        self._btn_in_file.clicked.connect(self._browse_input_file)
        self._btn_in_folder.clicked.connect(self._browse_input_folder)
        self._btn_out.clicked.connect(self._browse_output_folder)
        self._btn_clear_output.clicked.connect(self._clear_output_folder)
        self._btn_onnx.clicked.connect(self._browse_onnx_file)
        self._start_button.clicked.connect(self._on_start_clicked)
        self._cancel_button.clicked.connect(self._on_cancel_clicked)
        self._custom_res_button.clicked.connect(self._open_custom_res_dialog)

    # ========== Settings Persistence ==========

    def _load_settings(self) -> None:
        """Load settings from config file."""
        config = Config.load()

        self._onnx_edit.setText(config.onnx_path)
        self._tile_w_combo.setCurrentText(str(config.tile_w_limit))
        self._tile_h_combo.setCurrentText(str(config.tile_h_limit))
        self._model_scale_combo.setCurrentText(str(config.model_scale))

        self._same_dir_check.setChecked(config.same_dir)
        self._same_dir_suffix_edit.setText(config.same_dir_suffix)
        self._overwrite_check.setChecked(config.overwrite)
        self._alpha_check.setChecked(config.use_alpha)
        self._append_model_suffix_check.setChecked(config.append_model_suffix)

        self._fp16_check.setChecked(config.use_fp16)
        self._bf16_check.setChecked(config.use_bf16)
        self._tf32_check.setChecked(config.use_tf32)

        if config.input_path:
            p = Path(config.input_path)
            if p.exists():
                self._set_inputs_from_paths([p])

        self._custom_res_enabled = config.custom_res_enabled
        self._custom_res_width = config.custom_res_width
        self._custom_res_height = config.custom_res_height
        self._custom_res_maintain_ar = config.custom_res_maintain_ar

        self._secondary_enabled = config.secondary_enabled
        self._secondary_mode = config.secondary_mode
        self._secondary_width = config.secondary_width
        self._secondary_height = config.secondary_height

    def _save_settings(self) -> None:
        """Save current settings to config file."""
        first_input = str(self._input_items[0]) if self._input_items else ""

        config = Config(
            onnx_path=self._onnx_edit.text().strip(),
            tile_w_limit=self._parse_tile_value(self._tile_w_combo.currentText(), 1088),
            tile_h_limit=self._parse_tile_value(self._tile_h_combo.currentText(), 1920),
            model_scale=self._parse_model_scale(self._model_scale_combo.currentText()),
            same_dir=self._same_dir_check.isChecked(),
            same_dir_suffix=self._same_dir_suffix_edit.text(),
            append_model_suffix=self._append_model_suffix_check.isChecked(),
            overwrite=self._overwrite_check.isChecked(),
            use_alpha=self._alpha_check.isChecked(),
            use_fp16=self._fp16_check.isChecked(),
            use_bf16=self._bf16_check.isChecked(),
            use_tf32=self._tf32_check.isChecked(),
            input_path=first_input,
            custom_res_enabled=self._custom_res_enabled,
            custom_res_width=self._custom_res_width,
            custom_res_height=self._custom_res_height,
            custom_res_maintain_ar=self._custom_res_maintain_ar,
            secondary_enabled=self._secondary_enabled,
            secondary_mode=self._secondary_mode,
            secondary_width=self._secondary_width,
            secondary_height=self._secondary_height,
        )
        config.save()

    def closeEvent(self, event) -> None:
        """Handle window close event."""
        self._save_settings()
        cleanup_gui_input_tmp()
        super().closeEvent(event)

    # ========== Event Filter ==========

    def eventFilter(self, obj, event) -> bool:
        """Handle keyboard events for input field."""
        if obj is self._input_edit and event.type() == QEvent.KeyPress:
            if event.key() == Qt.Key_V and (event.modifiers() & Qt.ControlModifier):
                self._handle_clipboard_paste()
                return True
        return super().eventFilter(obj, event)

    # ========== Drag & Drop ==========

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:
        """Accept drag events for images, URLs, and text."""
        md = event.mimeData()
        if md.hasImage() or md.hasUrls() or md.hasText():
            event.acceptProposedAction()
        else:
            event.ignore()

    def dropEvent(self, event: QDropEvent) -> None:
        """Handle dropped content."""
        md = event.mimeData()

        # Raw image (e.g., from Discord)
        if md.hasImage():
            image = md.imageData()
            if isinstance(image, QImage) and not image.isNull():
                try:
                    GUI_INPUT_TMP_ROOT.mkdir(parents=True, exist_ok=True)
                    tmp_path = GUI_INPUT_TMP_ROOT / f"dragdrop-image-{int(time.time()*1000)}.png"
                    image.save(str(tmp_path), "PNG")
                    self._set_inputs_from_paths([tmp_path], default_output_for_clipboard=True)
                    return
                except Exception as e:
                    print(f"Drag-drop image error: {e}")

        # URLs (file or http/https)
        urls = md.urls()
        if urls:
            paths: list[Path] = []
            for url in urls:
                if url.isLocalFile():
                    p = Path(url.toLocalFile())
                    if p.exists():
                        paths.append(p)
                else:
                    s = url.toString()
                    if s.lower().startswith(("http://", "https://")):
                        p = self._download_url_to_temp(s)
                        if p is not None:
                            paths.append(p)

            if paths:
                self._set_inputs_from_paths(paths, default_output_for_clipboard=True)
                return

        # Plain text URL
        text = md.text().strip()
        if text.lower().startswith(("http://", "https://")):
            p = self._download_url_to_temp(text)
            if p is not None:
                self._set_inputs_from_paths([p], default_output_for_clipboard=True)
        else:
            event.ignore()

    # ========== Clipboard ==========

    def _handle_clipboard_paste(self) -> None:
        """Handle Ctrl+V paste in input field."""
        cb = QGuiApplication.clipboard()
        md = cb.mimeData()

        # Image directly in clipboard
        img = cb.image()
        if not img.isNull():
            try:
                GUI_INPUT_TMP_ROOT.mkdir(parents=True, exist_ok=True)
                tmp_path = GUI_INPUT_TMP_ROOT / f"clipboard-image-{int(time.time()*1000)}.png"
                img.save(str(tmp_path), "PNG")
                self._set_inputs_from_paths([tmp_path], default_output_for_clipboard=True)
            except Exception as e:
                print(f"Clipboard image error: {e}")
            return

        # URLs
        urls = md.urls()
        if urls:
            u = urls[0]
            if u.isLocalFile():
                p = Path(u.toLocalFile())
                if p.exists():
                    self._set_inputs_from_paths([p], default_output_for_clipboard=True)
                    return
            else:
                s = u.toString()
                if s.lower().startswith(("http://", "https://")):
                    p = self._download_url_to_temp(s)
                    if p is not None:
                        self._set_inputs_from_paths([p], default_output_for_clipboard=True)
                        return

        # Plain text URL
        text = md.text().strip()
        if text.lower().startswith(("http://", "https://")):
            p = self._download_url_to_temp(text)
            if p is not None:
                self._set_inputs_from_paths([p], default_output_for_clipboard=True)

    # ========== URL Helper ==========

    def _download_url_to_temp(self, url: str) -> Path | None:
        """Download a URL to the temporary input folder."""
        url = url.strip()
        if not url:
            return None

        try:
            parsed = urlparse(url)
            path = parsed.path or ""
            ext = os.path.splitext(path)[1]
            valid_exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
            if ext.lower() not in valid_exts:
                ext = ".png"

            GUI_INPUT_TMP_ROOT.mkdir(parents=True, exist_ok=True)

            base_name = os.path.basename(path)
            if not base_name:
                base_name = "dragdrop-image" + ext
            elif not os.path.splitext(base_name)[1]:
                base_name = base_name + ext

            dest = GUI_INPUT_TMP_ROOT / base_name
            urlretrieve(url, dest)
            return dest
        except Exception as e:
            print(f"Drag-drop URL error: {e}")
            return None

    # ========== Input/Output Helpers ==========

    def _set_inputs_from_paths(
        self,
        paths: list[Path],
        default_output_for_clipboard: bool = False,
    ) -> None:
        """Set input paths and update UI."""
        # Deduplicate
        seen: set[str] = set()
        unique: list[Path] = []
        for p in paths:
            s = str(p)
            if s not in seen:
                seen.add(s)
                unique.append(p)

        self._input_items = unique

        if not unique:
            self._input_edit.clear()
            self._current_file_label.setText("Current file: (none)")
            self._thumbnail_label.setText("(no thumbnail)")
            self._thumbnail_label.setPixmap(QPixmap())
            return

        # Find first file for thumbnail
        thumb_path: Path | None = None
        for p in unique:
            if p.is_file():
                thumb_path = p
                break

        # Update UI
        if len(unique) == 1:
            p = unique[0]
            self._input_edit.setText(str(p))
            if p.is_file() and thumb_path:
                self._update_thumbnail(str(thumb_path))
                self._current_file_label.setText(f"Current file: {p.name}")
            elif p.is_dir():
                self._thumbnail_label.setText("(folder)")
                self._thumbnail_label.setPixmap(QPixmap())
                self._current_file_label.setText(f"Current folder: {p}")
        else:
            self._input_edit.setText(f"{len(unique)} items")
            if thumb_path:
                self._update_thumbnail(str(thumb_path))
                self._current_file_label.setText(f"Current file: (multiple, first: {thumb_path.name})")
            else:
                self._thumbnail_label.setText("(multiple inputs)")
                self._thumbnail_label.setPixmap(QPixmap())
                self._current_file_label.setText("Current file: (multiple inputs)")

        # Set default output
        if default_output_for_clipboard:
            if not self._output_edit.text().strip():
                self._output_edit.setText(DEFAULT_OUTPUT_PATH)
        else:
            if not self._output_edit.text().strip():
                first = unique[0]
                base = first.parent if first.is_file() else first
                self._output_edit.setText(str(base / "upscaled"))

    def _browse_input_file(self) -> None:
        """Open file browser for input image."""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select input image",
            "",
            "Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.webp);;All files (*.*)",
        )
        if file_path:
            self._set_inputs_from_paths([Path(file_path)])

    def _browse_input_folder(self) -> None:
        """Open folder browser for input folder."""
        folder = QFileDialog.getExistingDirectory(self, "Select input folder")
        if folder:
            self._set_inputs_from_paths([Path(folder)])

    def _browse_output_folder(self) -> None:
        """Open folder browser for output folder."""
        folder = QFileDialog.getExistingDirectory(self, "Select output folder")
        if folder:
            self._output_edit.setText(folder)

    def _clear_output_folder(self) -> None:
        """Clear the output path."""
        self._output_edit.clear()

    def _browse_onnx_file(self) -> None:
        """Open file browser for ONNX model."""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select ONNX model",
            "",
            "ONNX files (*.onnx);;All files (*.*)",
        )
        if file_path:
            self._onnx_edit.setText(file_path)

    def _update_thumbnail(self, image_path: str) -> None:
        """Update the thumbnail preview."""
        pix = QPixmap(image_path)
        if pix.isNull():
            self._thumbnail_label.setText("(Failed to load thumbnail)")
            self._thumbnail_label.setPixmap(QPixmap())
            return

        scaled = pix.scaled(320, 320, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self._thumbnail_label.setPixmap(scaled)
        self._thumbnail_label.setText("")

        # Record original dimensions
        img = QImage(image_path)
        if not img.isNull():
            self._current_input_width = img.width()
            self._current_input_height = img.height()

    # ========== Custom Resolution Dialog ==========

    def _open_custom_res_dialog(self) -> None:
        """Open the custom resolution settings dialog."""
        dlg = CustomResolutionDialog(
            self,
            orig_width=self._current_input_width,
            orig_height=self._current_input_height,
            custom_enabled=self._custom_res_enabled,
            custom_width=self._custom_res_width,
            custom_height=self._custom_res_height,
            maintain_ar=self._custom_res_maintain_ar,
            secondary_enabled=self._secondary_enabled,
            secondary_mode=self._secondary_mode,
            secondary_width=self._secondary_width,
            secondary_height=self._secondary_height,
        )
        if dlg.exec() == QDialog.Accepted:
            settings = dlg.get_settings()
            self._custom_res_enabled = settings.custom_enabled
            self._custom_res_width = settings.custom_width
            self._custom_res_height = settings.custom_height
            self._custom_res_maintain_ar = settings.maintain_ar
            self._secondary_enabled = settings.secondary_enabled
            self._secondary_mode = settings.secondary_mode
            self._secondary_width = settings.secondary_width
            self._secondary_height = settings.secondary_height

    # ========== Validation ==========

    def _parse_tile_value(self, text: str, default: int) -> int:
        """Parse a tile dimension value."""
        try:
            return int(text.strip())
        except ValueError:
            return default

    def _validate_tile_value(self, text: str, name: str) -> int | None:
        """Validate that a tile value is a positive multiple of 64."""
        try:
            v = int(text.strip())
        except ValueError:
            QMessageBox.critical(self, "Error", f"{name} must be an integer.")
            return None
        if v <= 0 or v % 64 != 0:
            QMessageBox.critical(self, "Error", f"{name} must be a positive multiple of 64.")
            return None
        return v

    def _parse_model_scale(self, text: str) -> int:
        """Parse model scale value."""
        try:
            v = int(text.strip())
            if v in VALID_MODEL_SCALES:
                return v
        except ValueError:
            pass
        return 4

    def _validate_model_scale(self, text: str) -> int | None:
        """Validate model scale value."""
        try:
            v = int(text.strip())
        except ValueError:
            QMessageBox.critical(self, "Error", "Model scale must be 1, 2, 4, or 8.")
            return None
        if v not in VALID_MODEL_SCALES:
            QMessageBox.critical(self, "Error", "Model scale must be 1, 2, 4, or 8.")
            return None
        return v

    # ========== Start / Cancel ==========

    def _on_start_clicked(self) -> None:
        """Handle start button click."""
        if not self._input_items:
            QMessageBox.critical(self, "Error", "Please select at least one input file or folder.")
            return

        onnx_str = self._onnx_edit.text().strip()
        if not onnx_str:
            QMessageBox.critical(self, "Error", "Please select an ONNX model.")
            return
        if not Path(onnx_str).exists():
            QMessageBox.critical(self, "Error", f"ONNX model does not exist:\n{onnx_str}")
            return

        # Validate tile sizes
        tile_w = self._validate_tile_value(self._tile_w_combo.currentText(), "Tile W")
        if tile_w is None:
            return
        tile_h = self._validate_tile_value(self._tile_h_combo.currentText(), "Tile H")
        if tile_h is None:
            return

        model_scale = self._validate_model_scale(self._model_scale_combo.currentText())
        if model_scale is None:
            return

        # Validate custom resolution
        if self._custom_res_enabled:
            if self._custom_res_width <= 0 or self._custom_res_height <= 0:
                QMessageBox.critical(self, "Error", "Custom resolution width/height must be positive.")
                return

        # Get output directory
        output_str = self._output_edit.text().strip()
        if output_str:
            output_dir = Path(output_str)
        else:
            output_dir = Path(DEFAULT_OUTPUT_PATH)
            try:
                output_dir.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to create default output folder:\n{e}")
                return
            self._output_edit.setText(str(output_dir))

        self._save_settings()

        # Build file list
        exts = SUPPORTED_IMAGE_EXTENSIONS
        files: list[Path] = []
        for root in self._input_items:
            if root.is_file():
                files.append(root)
            else:
                files.extend(
                    p for p in sorted(root.rglob("*"))
                    if p.is_file() and p.suffix.lower() in exts
                )

        if not files:
            QMessageBox.critical(self, "Error", "No supported image files found.")
            return

        single_input_is_file = len(self._input_items) == 1 and self._input_items[0].is_file()

        # Secondary base dir
        if single_input_is_file:
            secondary_output_dir = output_dir.parent / "secondary-resized"
        else:
            first_root = self._input_items[0]
            base = first_root.parent if first_root.is_file() else first_root
            secondary_output_dir = base / "secondary-resized"

        # Ensure directories
        try:
            if not self._same_dir_check.isChecked():
                output_dir.mkdir(parents=True, exist_ok=True)
            if self._secondary_enabled:
                secondary_output_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to create output folders:\n{e}")
            return

        # Prepare UI for processing
        self._total_files_in_batch = len(files)
        self._completed_files_in_batch = 0
        self._current_avg_per_image = 0.0
        self._batch_start_time = time.perf_counter()

        if self._elapsed_timer is None:
            self._elapsed_timer = QTimer(self)
            self._elapsed_timer.timeout.connect(self._update_time_display)
        self._elapsed_timer.start(1000)

        self._progress_bar.setValue(0)
        self._progress_label.setText(f"Processing {len(files)} image(s)...")
        self._time_label.setText("Elapsed: 00:00:00 | ETA: --:--:--")
        self._avg_label.setText("Avg per image: -")
        self._current_file_label.setText("Current file: (starting...)")
        self._start_button.setEnabled(False)
        self._cancel_button.setEnabled(True)

        # Launch worker thread
        self._worker_thread = UpscaleWorkerThread(
            files=files,
            output_dir=output_dir,
            secondary_output_dir=secondary_output_dir,
            single_input_is_file=single_input_is_file,
            custom_res_enabled=self._custom_res_enabled,
            custom_width=self._custom_res_width,
            custom_height=self._custom_res_height,
            secondary_enabled=self._secondary_enabled,
            secondary_mode=self._secondary_mode,
            secondary_width=self._secondary_width,
            secondary_height=self._secondary_height,
            same_dir_enabled=self._same_dir_check.isChecked(),
            same_dir_suffix=self._same_dir_suffix_edit.text(),
            overwrite_enabled=self._overwrite_check.isChecked(),
            onnx_path=onnx_str,
            tile_w=str(tile_w),
            tile_h=str(tile_h),
            model_scale=str(model_scale),
            use_fp16=self._fp16_check.isChecked(),
            use_bf16=self._bf16_check.isChecked(),
            use_tf32=self._tf32_check.isChecked(),
            use_alpha=self._alpha_check.isChecked(),
            append_model_suffix_enabled=self._append_model_suffix_check.isChecked(),
        )
        self._worker_thread.progress_signal.connect(self._on_progress_update)
        self._worker_thread.thumbnail_signal.connect(self._on_thumbnail_update)
        self._worker_thread.finished_signal.connect(self._on_worker_finished)
        self._worker_thread.start()

    def _on_cancel_clicked(self) -> None:
        """Handle cancel button click."""
        if self._worker_thread is not None:
            self._worker_thread.cancel()
            self._progress_label.setText("Cancelling after current image...")

    def _update_time_display(self) -> None:
        """Update elapsed time and ETA labels."""
        if self._batch_start_time <= 0 or self._total_files_in_batch <= 0:
            self._time_label.setText("Elapsed: 00:00:00 | ETA: --:--:--")
            return

        elapsed = time.perf_counter() - self._batch_start_time
        elapsed_str = format_time_hms(elapsed)

        if self._current_avg_per_image > 0:
            remaining = max(self._total_files_in_batch - self._completed_files_in_batch, 0)
            eta_sec = remaining * self._current_avg_per_image
            eta_str = format_time_hms(eta_sec)
        else:
            eta_str = "--:--:--"

        self._time_label.setText(f"Elapsed: {elapsed_str} | ETA: {eta_str}")

    @Slot(int, int, float, str, str)
    def _on_progress_update(self, idx: int, total: int, avg: float, name: str, path: str) -> None:
        """Handle progress update from worker thread."""
        percent = int(100 * idx / total) if total > 0 else 0
        self._progress_bar.setValue(percent)
        self._progress_label.setText(f"Processed {idx}/{total} image(s)")
        self._completed_files_in_batch = idx
        self._current_avg_per_image = avg

        if avg > 0:
            self._avg_label.setText(f"Avg per image: {avg:.2f} s")

        self._current_file_label.setText(f"Current file: {name}")
        self._update_time_display()

    @Slot(str)
    def _on_thumbnail_update(self, image_path: str) -> None:
        """Handle thumbnail update from worker thread."""
        self._update_thumbnail(image_path)

    @Slot(str)
    def _on_worker_finished(self, final_text: str) -> None:
        """Handle worker thread completion."""
        self._progress_label.setText(final_text)
        self._start_button.setEnabled(True)
        self._cancel_button.setEnabled(False)

        if self._elapsed_timer is not None:
            self._elapsed_timer.stop()
            self._update_time_display()

        self._worker_thread = None
        cleanup_gui_input_tmp()
