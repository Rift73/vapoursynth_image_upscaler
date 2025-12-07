"""
Main application window for VapourSynth Image Upscaler.
"""

from __future__ import annotations

import ctypes
import os
import re
import sys
import time
from pathlib import Path

# Set Windows AppUserModelID for proper taskbar grouping/pinning
# Must be called before QApplication is created
if sys.platform == "win32":
    APP_ID = "VapourSynthImageUpscaler.GUI.1.0"
    ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(APP_ID)
from urllib.parse import urlparse
from urllib.request import urlretrieve, Request, urlopen

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
from PySide6.QtGui import QPixmap, QGuiApplication, QDragEnterEvent, QDropEvent, QImage, QIcon, QShortcut, QKeySequence
from PySide6.QtCore import Qt, QSize, QEvent, QTimer, Slot, QMimeData, QUrl

from ..core.constants import (
    GUI_INPUT_TMP_ROOT,
    SUPPORTED_IMAGE_EXTENSIONS,
    VALID_MODEL_SCALES,
    DEFAULT_OUTPUT_PATH,
    DEFAULT_ONNX_PATH,
    TEMP_BASE,
)
from ..core.config import Config
from ..core.utils import cleanup_gui_input_tmp, format_time_hms
from .dialogs import CustomResolutionDialog, AnimatedOutputDialog
from .dependencies_window import DependenciesWindow
from .worker_thread import UpscaleWorkerThread, ClipboardWorkerThread
from .theme import ThemeManager, AVAILABLE_THEMES

# Icon path - check multiple locations for PyInstaller compatibility
def _get_icon_path() -> Path | None:
    """Get icon path, checking multiple locations for PyInstaller compatibility."""
    # Check relative to this file (development)
    dev_path = Path(__file__).parent.parent.parent.parent / "icon.png"
    if dev_path.exists():
        return dev_path
    # Check next to exe (PyInstaller onefile)
    if getattr(sys, "frozen", False):
        exe_dir = Path(sys.executable).parent
        for name in ("icon.png", "icon.ico"):
            icon_path = exe_dir / name
            if icon_path.exists():
                return icon_path
        # Check _MEIPASS (PyInstaller temp folder)
        meipass = getattr(sys, "_MEIPASS", None)
        if meipass:
            for name in ("icon.png", "icon.ico"):
                icon_path = Path(meipass) / name
                if icon_path.exists():
                    return icon_path
    return None

_ICON_PATH = _get_icon_path()


def _natural_sort_key(path: Path) -> list:
    """
    Generate a sort key for natural sorting of paths.

    Splits the path string into text and numeric parts so that
    "file2.png" comes before "file10.png".

    Args:
        path: Path to generate sort key for.

    Returns:
        List of alternating strings and integers for comparison.
    """
    parts = re.split(r'(\d+)', str(path))
    return [int(p) if p.isdigit() else p.lower() for p in parts]


def parse_model_scale_from_filename(filename: str) -> int | None:
    """
    Parse model scale from ONNX filename.

    Looks for patterns like 1x, 2x, 4x, 8x or x1, x2, x4, x8 (case insensitive).

    Args:
        filename: The ONNX filename (without path).

    Returns:
        Scale value (1, 2, 4, or 8) if found, None otherwise.
    """
    # Pattern matches: 1x, 2x, 4x, 8x, x1, x2, x4, x8 (case insensitive)
    # Also handles patterns like _4x_, -4x-, 4x_, _x4, etc.
    patterns = [
        r'(?:^|[_\-\s])([1248])x(?:[_\-\s]|$)',  # 4x at boundaries
        r'(?:^|[_\-\s])x([1248])(?:[_\-\s]|$)',  # x4 at boundaries
        r'([1248])x(?=[_\-\.\s]|$)',              # 4x followed by separator or end
        r'x([1248])(?=[_\-\.\s]|$)',              # x4 followed by separator or end
    ]

    filename_lower = filename.lower()

    for pattern in patterns:
        match = re.search(pattern, filename_lower, re.IGNORECASE)
        if match:
            scale = int(match.group(1))
            if scale in (1, 2, 4, 8):
                return scale

    return None


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

        # Set window icon
        if _ICON_PATH and _ICON_PATH.exists():
            self.setWindowIcon(QIcon(str(_ICON_PATH)))

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
        self._current_image_start_time: float = 0.0  # When current image started

        # Custom resolution state
        self._custom_res_enabled = False
        self._custom_res_width = 0
        self._custom_res_height = 0
        self._custom_res_maintain_ar = True
        self._custom_res_mode = "width"
        self._custom_res_kernel = "lanczos"
        # Secondary output state
        self._secondary_enabled = False
        self._secondary_mode = "width"
        self._secondary_width = 1920
        self._secondary_height = 1080
        self._secondary_kernel = "lanczos"
        # Pre-scaling state
        self._prescale_enabled = False
        self._prescale_mode = "width"
        self._prescale_width = 1920
        self._prescale_height = 1080
        self._prescale_kernel = "lanczos"

        # Animated output state
        self._animated_output_format = "GIF"
        # GIF settings (gifski)
        self._gif_quality = 90
        self._gif_fast = False
        # WebP settings
        self._webp_quality = 90
        self._webp_lossless = True
        self._webp_preset = "none"
        # AVIF settings
        self._avif_quality = 80
        self._avif_quality_alpha = 90
        self._avif_speed = 6
        self._avif_lossless = False
        # APNG settings
        self._apng_pred = "mixed"

        # Model scale (auto-detected from ONNX filename or user-selected)
        self._model_scale: int = 4

        # Track if current inputs are from temp paths (clipboard/browser)
        self._inputs_from_temp: bool = False

        # Worker thread references
        self._worker_thread: UpscaleWorkerThread | None = None
        self._clipboard_worker: ClipboardWorkerThread | None = None

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

        # Tile combos
        self._tile_w_combo = QComboBox()
        self._tile_h_combo = QComboBox()

        for combo in (self._tile_w_combo, self._tile_h_combo):
            combo.setEditable(True)
            combo.addItems(["512", "768", "1024", "1088", "1536", "1920"])
            combo.setFixedWidth(100)

        self._tile_w_combo.setCurrentText("1088")
        self._tile_h_combo.setCurrentText("1920")

        # Batch size input (num_streams for Backend.TRT)
        self._batch_size_edit = QLineEdit("1")
        self._batch_size_edit.setFixedWidth(50)
        self._batch_size_edit.setToolTip("num_streams for TensorRT backend")

        # Checkboxes
        self._same_dir_check = QCheckBox("Save next to input with suffix:")
        self._same_dir_suffix_edit = QLineEdit("_upscaled")
        self._manga_folder_check = QCheckBox("Manga folder")
        self._manga_folder_check.setToolTip("Output: ParentFolder_suffix/Subfolder/.../Image.png")
        self._append_model_suffix_check = QCheckBox("Append model suffix")
        self._overwrite_check = QCheckBox("Overwrite")
        self._overwrite_check.setChecked(True)

        # Precision checkboxes
        self._fp16_check = QCheckBox("fp16")
        self._bf16_check = QCheckBox("bf16")
        self._bf16_check.setChecked(True)
        self._tf32_check = QCheckBox("tf32")

        # Sharpening widgets
        self._sharpen_check = QCheckBox("Sharpen")
        self._sharpen_value_edit = QLineEdit("0.50")
        self._sharpen_value_edit.setFixedWidth(50)
        self._sharpen_value_edit.setEnabled(False)

        # Labels
        self._current_file_label = QLabel("Current file: (none)")
        self._progress_label = QLabel("Idle")
        self._time_label = QLabel("Elapsed: 00:00:00 | ETA: --:--:--")
        self._avg_label = QLabel("Avg per image: -")
        self._progress_bar = QProgressBar()
        self._progress_bar.setRange(0, 100)
        self._thumbnail_label = QLabel("(no thumbnail)")
        self._thumbnail_label.setAlignment(Qt.AlignCenter)
        self._thumbnail_label.setMinimumSize(QSize(400, 200))
        self._image_info_label = QLabel("")
        self._image_info_label.setAlignment(Qt.AlignCenter)

        # Buttons
        self._btn_in_file = QPushButton("Browse File")
        self._btn_in_folder = QPushButton("Browse Folder")
        self._btn_out = QPushButton("Browse")
        self._btn_clear_output = QPushButton("Clear")
        self._btn_onnx = QPushButton("Browse ONNX")
        self._upscale_check = QCheckBox("Upscale")
        self._upscale_check.setChecked(True)  # Always default to enabled
        self._upscale_check.setToolTip("Enable SR upscaling. When disabled, only applies resolution/alpha processing.")
        self._start_button = QPushButton("Start")
        self._cancel_button = QPushButton("Cancel")
        self._cancel_button.setEnabled(False)
        self._clipboard_button = QPushButton("To Clipboard")
        self._clipboard_button.setToolTip("Upscale and copy result to clipboard (single image only)")
        self._custom_res_button = QPushButton("Resolution")
        self._animated_output_button = QPushButton("Animated Output")
        self._animated_output_button.setToolTip("Configure output format for animated content (GIF, WebP, AVIF)")
        self._dependencies_button = QPushButton("Dependencies")
        self._dependencies_button.setToolTip("Install required dependencies (VapourSynth plugins, ffmpeg, etc.)")

        # Theme dropdown
        self._theme_combo = QComboBox()
        self._theme_combo.addItems(AVAILABLE_THEMES)
        self._theme_combo.setToolTip("Select UI theme")
        self._theme_combo.setFixedWidth(80)

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
        main_layout.addWidget(self._upscale_check, row, 3)
        row += 1

        # Tile group
        tile_box = QGroupBox("Tile")
        tile_layout = QHBoxLayout()
        tile_layout.setSpacing(4)
        tile_box.setLayout(tile_layout)
        tile_layout.addWidget(QLabel("Width:"))
        tile_layout.addWidget(self._tile_w_combo)
        tile_layout.addSpacing(10)
        tile_layout.addWidget(QLabel("Height:"))
        tile_layout.addWidget(self._tile_h_combo)
        tile_layout.addSpacing(10)
        tile_layout.addWidget(QLabel("Batch size:"))
        tile_layout.addWidget(self._batch_size_edit)
        tile_layout.addStretch()
        main_layout.addWidget(tile_box, row, 0, 1, 4)
        row += 1

        # Precision options and sharpening in one row
        options_row = QHBoxLayout()

        prec_box = QGroupBox("vsmlrt")
        prec_layout = QHBoxLayout()
        prec_box.setLayout(prec_layout)
        prec_layout.addWidget(self._fp16_check)
        prec_layout.addWidget(self._bf16_check)
        prec_layout.addWidget(self._tf32_check)
        options_row.addWidget(prec_box)

        sharpen_box = QGroupBox("Sharpen")
        sharpen_layout = QHBoxLayout()
        sharpen_box.setLayout(sharpen_layout)
        sharpen_layout.addWidget(self._sharpen_check)
        sharpen_layout.addWidget(self._sharpen_value_edit)
        sharpen_layout.addWidget(QLabel("(0-1)"))
        options_row.addWidget(sharpen_box)

        options_row.addStretch()

        options_container = QWidget()
        options_container.setLayout(options_row)
        main_layout.addWidget(options_container, row, 0, 1, 4)
        row += 1

        # Same-dir + suffix + manga folder
        same_dir_layout = QHBoxLayout()
        same_dir_container = QWidget()
        same_dir_container.setLayout(same_dir_layout)
        same_dir_layout.addWidget(self._same_dir_check)
        same_dir_layout.addWidget(self._same_dir_suffix_edit)
        same_dir_layout.addWidget(self._manga_folder_check)
        same_dir_layout.addStretch()
        main_layout.addWidget(same_dir_container, row, 0, 1, 4)
        row += 1

        # Options row
        opts_layout = QHBoxLayout()
        opts_container = QWidget()
        opts_container.setLayout(opts_layout)
        opts_layout.addWidget(self._overwrite_check)
        opts_layout.addWidget(self._append_model_suffix_check)
        opts_layout.addWidget(self._custom_res_button)
        opts_layout.addWidget(self._animated_output_button)
        opts_layout.addWidget(self._dependencies_button)
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
        thumb_layout.addWidget(self._image_info_label)
        main_layout.addWidget(thumb_box, 0, 4, row, 1)

        # Buttons at bottom (Start, Cancel, To Clipboard, Theme)
        btn_layout = QHBoxLayout()
        btn_container = QWidget()
        btn_container.setLayout(btn_layout)
        btn_layout.addWidget(self._start_button, 2)
        btn_layout.addWidget(self._cancel_button, 2)
        btn_layout.addWidget(self._clipboard_button, 1)
        btn_layout.addWidget(QLabel("Theme:"))
        btn_layout.addWidget(self._theme_combo)
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
        self._clipboard_button.clicked.connect(self._on_clipboard_clicked)
        self._custom_res_button.clicked.connect(self._open_custom_res_dialog)
        self._animated_output_button.clicked.connect(self._open_animated_output_dialog)
        self._dependencies_button.clicked.connect(self._open_dependencies_window)
        self._sharpen_check.toggled.connect(self._on_sharpen_toggled)
        self._manga_folder_check.toggled.connect(self._on_manga_folder_toggled)
        self._upscale_check.toggled.connect(self._on_upscale_toggled)
        self._theme_combo.currentTextChanged.connect(self._on_theme_changed)

        # Keyboard shortcuts
        self._open_log_shortcut = QShortcut(QKeySequence("Ctrl+L"), self)
        self._open_log_shortcut.activated.connect(self._on_open_log_clicked)

    # ========== Settings Persistence ==========

    def _load_settings(self) -> None:
        """Load settings from config file."""
        config = Config.load()

        self._onnx_edit.setText(config.onnx_path)
        self._tile_w_combo.setCurrentText(str(config.tile_w_limit))
        self._tile_h_combo.setCurrentText(str(config.tile_h_limit))
        self._model_scale = config.model_scale

        # Try to auto-detect model scale from ONNX filename
        if config.onnx_path:
            detected_scale = parse_model_scale_from_filename(Path(config.onnx_path).name)
            if detected_scale is not None:
                self._model_scale = detected_scale

        self._same_dir_check.setChecked(config.same_dir)
        self._same_dir_suffix_edit.setText(config.same_dir_suffix)
        self._manga_folder_check.setChecked(config.manga_folder)
        # Apply manga folder toggle effect on load
        if config.manga_folder:
            self._same_dir_check.setEnabled(False)
        self._overwrite_check.setChecked(config.overwrite)
        self._append_model_suffix_check.setChecked(config.append_model_suffix)

        self._fp16_check.setChecked(config.use_fp16)
        self._bf16_check.setChecked(config.use_bf16)
        self._tf32_check.setChecked(config.use_tf32)
        self._batch_size_edit.setText(str(config.num_streams))

        if config.input_path:
            p = Path(config.input_path)
            if p.exists():
                self._set_inputs_from_paths([p])

        self._custom_res_enabled = config.custom_res_enabled
        self._custom_res_width = config.custom_res_width
        self._custom_res_height = config.custom_res_height
        self._custom_res_maintain_ar = config.custom_res_maintain_ar
        self._custom_res_mode = config.custom_res_mode
        self._custom_res_kernel = config.custom_res_kernel

        self._secondary_enabled = config.secondary_enabled
        self._secondary_mode = config.secondary_mode
        self._secondary_width = config.secondary_width
        self._secondary_height = config.secondary_height
        self._secondary_kernel = config.secondary_kernel

        self._prescale_enabled = config.prescale_enabled
        self._prescale_mode = config.prescale_mode
        self._prescale_width = config.prescale_width
        self._prescale_height = config.prescale_height
        self._prescale_kernel = config.prescale_kernel

        # Load sharpen settings into widgets
        self._sharpen_check.setChecked(config.sharpen_enabled)
        self._sharpen_value_edit.setText(f"{config.sharpen_value:.2f}")
        self._sharpen_value_edit.setEnabled(config.sharpen_enabled)

        # Load animated output settings
        self._animated_output_format = config.animated_output_format
        self._gif_quality = config.gif_quality
        self._gif_fast = config.gif_fast
        self._webp_quality = config.webp_quality
        self._webp_lossless = config.webp_lossless
        self._webp_preset = config.webp_preset
        self._avif_quality = config.avif_quality
        self._avif_quality_alpha = config.avif_quality_alpha
        self._avif_speed = config.avif_speed
        self._avif_lossless = config.avif_lossless
        self._apng_pred = config.apng_pred

        # Load theme and apply (without triggering save)
        if config.theme in AVAILABLE_THEMES:
            self._theme_combo.blockSignals(True)
            self._theme_combo.setCurrentText(config.theme)
            self._theme_combo.blockSignals(False)
            ThemeManager.apply_theme(config.theme)

    def _save_settings(self) -> None:
        """Save current settings to config file."""
        first_input = str(self._input_items[0]) if self._input_items else ""

        config = Config(
            onnx_path=self._onnx_edit.text().strip(),
            tile_w_limit=self._parse_tile_value(self._tile_w_combo.currentText(), 1088),
            tile_h_limit=self._parse_tile_value(self._tile_h_combo.currentText(), 1920),
            model_scale=self._model_scale,
            same_dir=self._same_dir_check.isChecked(),
            same_dir_suffix=self._same_dir_suffix_edit.text(),
            manga_folder=self._manga_folder_check.isChecked(),
            append_model_suffix=self._append_model_suffix_check.isChecked(),
            overwrite=self._overwrite_check.isChecked(),
            use_fp16=self._fp16_check.isChecked(),
            use_bf16=self._bf16_check.isChecked(),
            use_tf32=self._tf32_check.isChecked(),
            num_streams=self._parse_batch_size(self._batch_size_edit.text()),
            input_path=first_input,
            custom_res_enabled=self._custom_res_enabled,
            custom_res_width=self._custom_res_width,
            custom_res_height=self._custom_res_height,
            custom_res_maintain_ar=self._custom_res_maintain_ar,
            custom_res_mode=self._custom_res_mode,
            custom_res_kernel=self._custom_res_kernel,
            secondary_enabled=self._secondary_enabled,
            secondary_mode=self._secondary_mode,
            secondary_width=self._secondary_width,
            secondary_height=self._secondary_height,
            secondary_kernel=self._secondary_kernel,
            prescale_enabled=self._prescale_enabled,
            prescale_mode=self._prescale_mode,
            prescale_width=self._prescale_width,
            prescale_height=self._prescale_height,
            prescale_kernel=self._prescale_kernel,
            sharpen_enabled=self._sharpen_check.isChecked(),
            sharpen_value=self._get_sharpen_value(),
            animated_output_format=self._animated_output_format,
            gif_quality=self._gif_quality,
            gif_fast=self._gif_fast,
            webp_quality=self._webp_quality,
            webp_lossless=self._webp_lossless,
            webp_preset=self._webp_preset,
            avif_quality=self._avif_quality,
            avif_quality_alpha=self._avif_quality_alpha,
            avif_speed=self._avif_speed,
            avif_lossless=self._avif_lossless,
            apng_pred=self._apng_pred,
            theme=self._theme_combo.currentText(),
        )
        config.save()

    def _on_sharpen_toggled(self, checked: bool) -> None:
        """Handle sharpen checkbox toggle."""
        self._sharpen_value_edit.setEnabled(checked)

    def _on_manga_folder_toggled(self, checked: bool) -> None:
        """Handle manga folder checkbox toggle - disables same_dir when enabled."""
        if checked:
            self._same_dir_check.setChecked(False)
            self._same_dir_check.setEnabled(False)
        else:
            self._same_dir_check.setEnabled(True)

    def _on_upscale_toggled(self, checked: bool) -> None:
        """Handle upscale checkbox toggle - disables pre-scale when upscaling is disabled."""
        # Pre-scale only makes sense when upscaling is enabled
        # When upscale is disabled, pre-scale should be greyed out
        # Note: The actual prescale_enabled state is managed in the Resolution dialog
        # Here we just visually indicate that pre-scale won't work without upscaling
        pass  # The Resolution dialog will check upscale state when opened

    def _on_theme_changed(self, theme_name: str) -> None:
        """Handle theme dropdown change."""
        ThemeManager.apply_theme(theme_name)
        self._save_settings()

    def _get_sharpen_value(self) -> float:
        """Get the current sharpen value from the text field."""
        try:
            value = float(self._sharpen_value_edit.text().strip())
            return max(0.0, min(1.0, value))
        except ValueError:
            return 0.5

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
        """Accept drag events for images, URLs, HTML, and text."""
        md = event.mimeData()
        if md.hasImage() or md.hasUrls() or md.hasHtml() or md.hasText():
            event.acceptProposedAction()
        else:
            event.ignore()

    def dropEvent(self, event: QDropEvent) -> None:
        """Handle dropped content."""
        md = event.mimeData()

        # Raw image (e.g., from some applications)
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

        # Try to extract image URL from HTML content (Discord, etc.)
        if md.hasHtml():
            html = md.html()
            img_url = self._extract_image_url_from_html(html)
            if img_url:
                p = self._download_url_to_temp(img_url)
                if p is not None:
                    self._set_inputs_from_paths([p], default_output_for_clipboard=True)
                    return

        # Plain text - could be URL or contain URL
        text = md.text().strip() if md.hasText() else ""
        if text:
            # Check if it's a direct URL
            if text.lower().startswith(("http://", "https://")):
                p = self._download_url_to_temp(text)
                if p is not None:
                    self._set_inputs_from_paths([p], default_output_for_clipboard=True)
                    return
            # Try to find URL in text (Discord sometimes includes extra text)
            url_match = re.search(r'https?://[^\s<>"]+\.(?:png|jpg|jpeg|gif|webp|bmp|tiff?)', text, re.IGNORECASE)
            if url_match:
                p = self._download_url_to_temp(url_match.group(0))
                if p is not None:
                    self._set_inputs_from_paths([p], default_output_for_clipboard=True)
                    return

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

    def _extract_image_url_from_html(self, html: str) -> str | None:
        """
        Extract an image URL from HTML content.

        Discord and other sites often provide HTML with embedded image tags
        when dragging images. This extracts the src attribute from img tags.

        Args:
            html: HTML content from drag-drop mimeData.

        Returns:
            Image URL string or None if not found.
        """
        # Common Discord CDN patterns
        discord_patterns = [
            r'https://cdn\.discordapp\.com/attachments/[^\s"\'<>]+',
            r'https://media\.discordapp\.net/attachments/[^\s"\'<>]+',
            r'https://cdn\.discordapp\.com/ephemeral-attachments/[^\s"\'<>]+',
        ]

        # Try Discord-specific patterns first
        for pattern in discord_patterns:
            match = re.search(pattern, html, re.IGNORECASE)
            if match:
                url = match.group(0)
                # Clean up any trailing characters that might have been captured
                url = re.sub(r'["\'\s<>].*$', '', url)
                return url

        # Generic img src extraction
        img_patterns = [
            r'<img[^>]+src=["\']([^"\']+)["\']',
            r'<img[^>]+src=([^\s>]+)',
        ]

        for pattern in img_patterns:
            match = re.search(pattern, html, re.IGNORECASE)
            if match:
                url = match.group(1)
                # Check if it looks like an image URL
                if re.search(r'\.(png|jpg|jpeg|gif|webp|bmp|tiff?)(\?|$)', url, re.IGNORECASE):
                    return url
                # Discord CDN URLs may not have extensions
                if 'discordapp' in url or 'discord' in url:
                    return url

        # Try to find any image URL in the HTML
        general_img_pattern = r'https?://[^\s"\'<>]+\.(?:png|jpg|jpeg|gif|webp|bmp|tiff?)(?:\?[^\s"\'<>]*)?'
        match = re.search(general_img_pattern, html, re.IGNORECASE)
        if match:
            return match.group(0)

        return None

    def _download_url_to_temp(self, url: str) -> Path | None:
        """
        Download a URL to the temporary input folder.

        Uses proper headers to handle Discord CDN and other sites that
        may block requests without a User-Agent.

        Args:
            url: The URL to download.

        Returns:
            Path to the downloaded file, or None on failure.
        """
        url = url.strip()
        if not url:
            return None

        try:
            parsed = urlparse(url)
            path = parsed.path or ""
            ext = os.path.splitext(path)[1]
            valid_exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp", ".gif"}
            if ext.lower() not in valid_exts:
                ext = ".png"

            GUI_INPUT_TMP_ROOT.mkdir(parents=True, exist_ok=True)

            base_name = os.path.basename(path)
            if not base_name:
                base_name = f"dragdrop-image-{int(time.time()*1000)}" + ext
            elif not os.path.splitext(base_name)[1]:
                base_name = base_name + ext

            # Ensure unique filename to avoid collisions
            dest = GUI_INPUT_TMP_ROOT / base_name
            if dest.exists():
                stem = dest.stem
                dest = GUI_INPUT_TMP_ROOT / f"{stem}_{int(time.time()*1000)}{ext}"

            # Use Request with headers for Discord CDN and other restrictive sites
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                "Accept": "image/webp,image/apng,image/*,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.9",
                "Referer": f"{parsed.scheme}://{parsed.netloc}/",
            }

            req = Request(url, headers=headers)
            with urlopen(req, timeout=30) as response:
                with open(dest, "wb") as f:
                    f.write(response.read())

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

        # Check if any inputs are from temp paths (clipboard/browser drag-drop)
        temp_root_str = str(GUI_INPUT_TMP_ROOT)
        any_from_temp = any(str(p).startswith(temp_root_str) for p in unique)

        # Track temp input state and update same_dir checkbox accordingly
        self._inputs_from_temp = any_from_temp
        if any_from_temp:
            # Disable "save next to input" for temp paths
            self._same_dir_check.setChecked(False)
            self._same_dir_check.setEnabled(False)
            self._same_dir_suffix_edit.setEnabled(False)
        else:
            # Re-enable for regular paths
            self._same_dir_check.setEnabled(True)
            self._same_dir_suffix_edit.setEnabled(True)

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
                self._image_info_label.setText("")
                self._current_file_label.setText(f"Current folder: {p}")
        else:
            self._input_edit.setText(f"{len(unique)} items")
            if thumb_path:
                self._update_thumbnail(str(thumb_path))
                self._current_file_label.setText(f"Current file: (multiple, first: {thumb_path.name})")
            else:
                self._thumbnail_label.setText("(multiple inputs)")
                self._thumbnail_label.setPixmap(QPixmap())
                self._image_info_label.setText("")
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
        # Determine starting directory:
        # 1. Use last browse directory if set and exists
        # 2. Otherwise use parent of current ONNX path if set and exists
        # 3. Otherwise use empty string (system default)
        start_dir = ""
        try:
            config = Config.load()
            if config.last_onnx_browse_dir:
                last_dir = Path(config.last_onnx_browse_dir)
                if last_dir.is_dir():
                    start_dir = str(last_dir)
        except Exception:
            pass

        if not start_dir:
            try:
                current_text = self._onnx_edit.text()
                if current_text:
                    current_parent = Path(current_text).parent
                    if current_parent.is_dir():
                        start_dir = str(current_parent)
            except Exception:
                pass

        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select ONNX model",
            start_dir,
            "ONNX files (*.onnx);;All files (*.*)",
        )
        if file_path:
            self._onnx_edit.setText(file_path)
            self._detect_or_ask_model_scale(file_path)
            # Save the directory for next time
            try:
                config = Config.load()
                config.last_onnx_browse_dir = str(Path(file_path).parent)
                config.save()
            except Exception:
                pass

    def _detect_or_ask_model_scale(self, onnx_path: str) -> None:
        """
        Detect model scale from ONNX filename or ask user to select.

        Args:
            onnx_path: Path to the ONNX model file.
        """
        filename = Path(onnx_path).name
        detected_scale = parse_model_scale_from_filename(filename)

        if detected_scale is not None:
            self._model_scale = detected_scale
            return

        # Scale not detected, show dialog
        scale = self._show_model_scale_dialog(filename)
        if scale is not None:
            self._model_scale = scale

    def _show_model_scale_dialog(self, filename: str) -> int | None:
        """
        Show a dialog to select model scale.

        Args:
            filename: ONNX filename for display in the dialog.

        Returns:
            Selected scale (1, 2, 4, or 8) or None if cancelled.
        """
        dialog = QDialog(self)
        dialog.setWindowTitle("Select Model Scale")
        dialog.setModal(True)

        layout = QVBoxLayout()
        dialog.setLayout(layout)

        # Info label
        info_label = QLabel(
            f"Could not detect model scale from filename:\n\n"
            f"  {filename}\n\n"
            f"Please select the scale factor for this model:"
        )
        layout.addWidget(info_label)

        # Dropdown for scale selection
        combo_layout = QHBoxLayout()
        combo_layout.addWidget(QLabel("Model Scale (SR):"))
        scale_combo = QComboBox()
        scale_combo.addItems(["1", "2", "4", "8"])
        scale_combo.setCurrentText("4")  # Default to 4x
        combo_layout.addWidget(scale_combo)
        combo_layout.addStretch()
        layout.addLayout(combo_layout)

        # Buttons
        btn_layout = QHBoxLayout()
        ok_btn = QPushButton("OK")
        cancel_btn = QPushButton("Cancel")
        btn_layout.addStretch()
        btn_layout.addWidget(ok_btn)
        btn_layout.addWidget(cancel_btn)
        layout.addLayout(btn_layout)

        result: list[int | None] = [None]

        def on_ok() -> None:
            result[0] = int(scale_combo.currentText())
            dialog.accept()

        def on_cancel() -> None:
            dialog.reject()

        ok_btn.clicked.connect(on_ok)
        cancel_btn.clicked.connect(on_cancel)

        dialog.exec()
        return result[0]

    def _update_thumbnail(self, image_path: str) -> None:
        """Update the thumbnail preview."""
        from pathlib import Path

        pix = QPixmap(image_path)
        if pix.isNull():
            self._thumbnail_label.setText("(Failed to load thumbnail)")
            self._thumbnail_label.setPixmap(QPixmap())
            self._image_info_label.setText("")
            return

        scaled = pix.scaled(640, 320, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self._thumbnail_label.setPixmap(scaled)
        self._thumbnail_label.setText("")

        # Record original dimensions
        img = QImage(image_path)
        if not img.isNull():
            self._current_input_width = img.width()
            self._current_input_height = img.height()

            # Update image info label with resolution and format
            file_ext = Path(image_path).suffix.upper().lstrip(".")
            if not file_ext:
                file_ext = "Unknown"
            self._image_info_label.setText(
                f"{img.width()} × {img.height()} {file_ext}"
            )
        else:
            self._image_info_label.setText("")

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
            custom_mode=self._custom_res_mode,
            custom_kernel=self._custom_res_kernel,
            secondary_enabled=self._secondary_enabled,
            secondary_mode=self._secondary_mode,
            secondary_width=self._secondary_width,
            secondary_height=self._secondary_height,
            secondary_kernel=self._secondary_kernel,
            prescale_enabled=self._prescale_enabled,
            prescale_mode=self._prescale_mode,
            prescale_width=self._prescale_width,
            prescale_height=self._prescale_height,
            prescale_kernel=self._prescale_kernel,
            upscale_enabled=self._upscale_check.isChecked(),
        )
        if dlg.exec() == QDialog.Accepted:
            settings = dlg.get_settings()
            self._custom_res_enabled = settings.custom_enabled
            self._custom_res_width = settings.custom_width
            self._custom_res_height = settings.custom_height
            self._custom_res_maintain_ar = settings.maintain_ar
            self._custom_res_mode = settings.custom_mode
            self._custom_res_kernel = settings.custom_kernel
            self._secondary_enabled = settings.secondary_enabled
            self._secondary_mode = settings.secondary_mode
            self._secondary_width = settings.secondary_width
            self._secondary_height = settings.secondary_height
            self._secondary_kernel = settings.secondary_kernel
            self._prescale_enabled = settings.prescale_enabled
            self._prescale_mode = settings.prescale_mode
            self._prescale_width = settings.prescale_width
            self._prescale_height = settings.prescale_height
            self._prescale_kernel = settings.prescale_kernel

    # ========== Animated Output Dialog ==========

    def _open_animated_output_dialog(self) -> None:
        """Open the animated output settings dialog."""
        try:
            dlg = AnimatedOutputDialog(
                self,
                output_format=self._animated_output_format,
                gif_quality=self._gif_quality,
                gif_fast=self._gif_fast,
                webp_quality=self._webp_quality,
                webp_lossless=self._webp_lossless,
                webp_preset=self._webp_preset,
                avif_quality=self._avif_quality,
                avif_quality_alpha=self._avif_quality_alpha,
                avif_speed=self._avif_speed,
                avif_lossless=self._avif_lossless,
                apng_pred=self._apng_pred,
            )
            if dlg.exec() == QDialog.Accepted:
                settings = dlg.get_settings()
                self._animated_output_format = settings.output_format
                self._gif_quality = settings.gif_quality
                self._gif_fast = settings.gif_fast
                self._webp_quality = settings.webp_quality
                self._webp_lossless = settings.webp_lossless
                self._webp_preset = settings.webp_preset
                self._avif_quality = settings.avif_quality
                self._avif_quality_alpha = settings.avif_quality_alpha
                self._avif_speed = settings.avif_speed
                self._avif_lossless = settings.avif_lossless
                self._apng_pred = settings.apng_pred
        except Exception as e:
            import traceback
            QMessageBox.critical(self, "Error", f"Failed to open dialog: {e}\n\n{traceback.format_exc()}")

    # ========== Dependencies Window ==========

    def _open_dependencies_window(self) -> None:
        """Open the dependencies installation window."""
        try:
            dlg = DependenciesWindow(self)
            dlg.exec()
        except Exception as e:
            import traceback
            QMessageBox.critical(self, "Error", f"Failed to open dependencies window: {e}\n\n{traceback.format_exc()}")

    # ========== Log File ==========

    def _on_open_log_clicked(self) -> None:
        """Open the worker debug log file in the default text editor."""
        log_file = TEMP_BASE / "worker_debug.log"
        if not log_file.exists():
            QMessageBox.information(
                self,
                "Log File",
                f"Log file does not exist yet.\nIt will be created after the first processing run.\n\nPath: {log_file}",
            )
            return

        # Open the log file with the system default application
        try:
            os.startfile(str(log_file))
        except Exception as e:
            QMessageBox.warning(
                self,
                "Error",
                f"Could not open log file:\n{e}\n\nPath: {log_file}",
            )

    # ========== Validation ==========

    def _parse_tile_value(self, text: str, default: int) -> int:
        """Parse a tile dimension value."""
        try:
            return int(text.strip())
        except ValueError:
            return default

    def _parse_batch_size(self, text: str) -> int:
        """Parse batch size (num_streams) value."""
        try:
            return max(1, int(text.strip()))
        except ValueError:
            return 1

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

    def _should_use_batch_mode(self, files: list[Path]) -> bool:
        """
        Determine if batch mode should be used based on file types.

        Batch mode is disabled when:
        - Any input file is an animated format (GIF)
        - Less than 2 files to process

        Note: Alpha is now auto-detected per-file by the worker thread.
        Files with alpha will be automatically separated from the batch.
        """
        # Need at least 2 files for batch mode to make sense
        if len(files) < 2:
            return False

        # Check for animated formats that need individual processing
        animated_extensions = {".gif"}
        for f in files:
            if f.suffix.lower() in animated_extensions:
                return False

        return True

    # ========== Start / Cancel ==========

    def _on_start_clicked(self) -> None:
        """Handle start button click."""
        if not self._input_items:
            QMessageBox.critical(self, "Error", "Please select at least one input file or folder.")
            return

        # Only validate ONNX and tile settings when upscaling is enabled
        upscale_enabled = self._upscale_check.isChecked()
        onnx_str = self._onnx_edit.text().strip()
        if upscale_enabled:
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
        else:
            # When upscaling is disabled, use default tile values (they won't be used anyway)
            tile_w = 1088
            tile_h = 1920

        # Use the auto-detected or user-selected model scale
        model_scale = self._model_scale

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
                    p for p in sorted(root.rglob("*"), key=_natural_sort_key)
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
        self._current_image_start_time = 0.0
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
            input_roots=list(self._input_items),
            custom_res_enabled=self._custom_res_enabled,
            custom_res_mode=self._custom_res_mode,
            custom_width=self._custom_res_width,
            custom_height=self._custom_res_height,
            custom_res_kernel=self._custom_res_kernel,
            secondary_enabled=self._secondary_enabled,
            secondary_mode=self._secondary_mode,
            secondary_width=self._secondary_width,
            secondary_height=self._secondary_height,
            secondary_kernel=self._secondary_kernel,
            same_dir_enabled=self._same_dir_check.isChecked(),
            same_dir_suffix=self._same_dir_suffix_edit.text(),
            manga_folder_enabled=self._manga_folder_check.isChecked(),
            overwrite_enabled=self._overwrite_check.isChecked(),
            onnx_path=onnx_str,
            tile_w=str(tile_w),
            tile_h=str(tile_h),
            model_scale=str(model_scale),
            use_fp16=self._fp16_check.isChecked(),
            use_bf16=self._bf16_check.isChecked(),
            use_tf32=self._tf32_check.isChecked(),
            num_streams=self._parse_batch_size(self._batch_size_edit.text()),
            append_model_suffix_enabled=self._append_model_suffix_check.isChecked(),
            prescale_enabled=self._prescale_enabled,
            prescale_mode=self._prescale_mode,
            prescale_width=self._prescale_width,
            prescale_height=self._prescale_height,
            prescale_kernel=self._prescale_kernel,
            sharpen_enabled=self._sharpen_check.isChecked(),
            sharpen_value=self._get_sharpen_value(),
            use_batch_mode=self._should_use_batch_mode(files),
            animated_output_format=self._animated_output_format,
            gif_quality=self._gif_quality,
            gif_fast=self._gif_fast,
            webp_quality=self._webp_quality,
            webp_lossless=self._webp_lossless,
            webp_preset=self._webp_preset,
            avif_quality=self._avif_quality,
            avif_quality_alpha=self._avif_quality_alpha,
            avif_speed=self._avif_speed,
            avif_lossless=self._avif_lossless,
            apng_pred=self._apng_pred,
            upscale_enabled=self._upscale_check.isChecked(),
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

    def _on_clipboard_clicked(self) -> None:
        """Handle clipboard button click - upscale single image and copy to clipboard."""
        # Must have exactly one input file (not folder)
        if not self._input_items:
            QMessageBox.critical(self, "Error", "Please select an input image first.")
            return

        if len(self._input_items) != 1:
            QMessageBox.critical(self, "Error", "Clipboard output only works with a single image.")
            return

        input_path = self._input_items[0]
        if not input_path.is_file():
            QMessageBox.critical(self, "Error", "Clipboard output requires a single file, not a folder.")
            return

        # Check extension
        if input_path.suffix.lower() not in SUPPORTED_IMAGE_EXTENSIONS:
            QMessageBox.critical(self, "Error", f"Unsupported image format: {input_path.suffix}")
            return

        # Validate ONNX
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

        # Use the internally tracked model scale (auto-detected or user-selected)
        model_scale = self._model_scale

        # Disable buttons during processing
        self._start_button.setEnabled(False)
        self._cancel_button.setEnabled(False)
        self._clipboard_button.setEnabled(False)
        self._progress_label.setText("Upscaling to clipboard...")
        self._progress_bar.setValue(0)

        # Launch clipboard worker
        self._clipboard_worker = ClipboardWorkerThread(
            input_file=input_path,
            onnx_path=onnx_str,
            tile_w=str(tile_w),
            tile_h=str(tile_h),
            model_scale=str(model_scale),
            use_fp16=self._fp16_check.isChecked(),
            use_bf16=self._bf16_check.isChecked(),
            use_tf32=self._tf32_check.isChecked(),
            custom_res_enabled=self._custom_res_enabled,
            custom_res_mode=self._custom_res_mode,
            custom_width=self._custom_res_width,
            custom_height=self._custom_res_height,
            custom_res_kernel=self._custom_res_kernel,
            prescale_enabled=self._prescale_enabled,
            prescale_mode=self._prescale_mode,
            prescale_width=self._prescale_width,
            prescale_height=self._prescale_height,
            prescale_kernel=self._prescale_kernel,
            sharpen_enabled=self._sharpen_check.isChecked(),
            sharpen_value=self._get_sharpen_value(),
        )
        self._clipboard_worker.status_signal.connect(self._on_clipboard_status)
        self._clipboard_worker.result_signal.connect(self._on_clipboard_result)
        self._clipboard_worker.start()

    @Slot(str)
    def _on_clipboard_status(self, status: str) -> None:
        """Handle status update from clipboard worker."""
        self._progress_label.setText(status)

    @Slot(str)
    def _on_clipboard_result(self, image_path: str) -> None:
        """Handle clipboard worker completion - copy image to clipboard."""
        self._start_button.setEnabled(True)
        self._clipboard_button.setEnabled(True)
        self._progress_bar.setValue(100)

        if image_path:
            # Load the image and copy to clipboard
            image = QImage(image_path)
            if not image.isNull():
                clipboard = QGuiApplication.clipboard()

                # Use QMimeData to set both the image and file URL
                # This provides better alpha channel support across applications
                mime_data = QMimeData()

                # Set the image data (for apps that accept image data)
                mime_data.setImageData(image)

                # Set the file URL (for apps that prefer file references - preserves alpha)
                file_url = QUrl.fromLocalFile(image_path)
                mime_data.setUrls([file_url])

                clipboard.setMimeData(mime_data)
                self._progress_label.setText("Image copied to clipboard!")
            else:
                self._progress_label.setText("Failed to load upscaled image.")
        else:
            self._progress_label.setText("Upscale failed.")

        self._clipboard_worker = None

    def _update_time_display(self) -> None:
        """Update elapsed time and ETA labels with smooth countdown."""
        if self._batch_start_time <= 0 or self._total_files_in_batch <= 0:
            self._time_label.setText("Elapsed: 00:00:00 | ETA: --:--:--")
            return

        now = time.perf_counter()
        elapsed = now - self._batch_start_time
        elapsed_str = format_time_hms(elapsed)

        if self._current_avg_per_image > 0:
            # Calculate remaining full images after the current one
            remaining_after_current = max(
                self._total_files_in_batch - self._completed_files_in_batch - 1, 0
            )

            # Time for remaining full images
            eta_remaining_images = remaining_after_current * self._current_avg_per_image

            # Time remaining on current image (smooth countdown)
            if self._current_image_start_time > 0:
                time_on_current = now - self._current_image_start_time
                eta_current_image = max(0, self._current_avg_per_image - time_on_current)
            else:
                # No current image started yet, use full average
                eta_current_image = self._current_avg_per_image

            eta_sec = eta_remaining_images + eta_current_image
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

        # Detect if a new image is starting (idx matches previous completed count)
        # This happens when we receive the "before processing" signal
        if idx == self._completed_files_in_batch and idx < total:
            # New image starting - reset the current image timer
            self._current_image_start_time = time.perf_counter()

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
