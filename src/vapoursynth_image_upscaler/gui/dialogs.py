"""
Dialog windows for the GUI.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QHBoxLayout,
    QGridLayout,
    QGroupBox,
    QLabel,
    QLineEdit,
    QPushButton,
    QCheckBox,
    QComboBox,
    QSlider,
    QSpinBox,
)

if TYPE_CHECKING:
    from PySide6.QtWidgets import QWidget


# Valid animated output formats
ANIMATED_OUTPUT_FORMATS = ["GIF", "WebP", "AVIF", "APNG"]

# WebP presets for FFmpeg libwebp encoder
WEBP_PRESETS = ["none", "default", "picture", "photo", "drawing", "icon", "text"]

# APNG prediction methods for FFmpeg apng encoder
# none=fastest, mixed=slowest but best compression
APNG_PRED_METHODS = ["none", "sub", "up", "avg", "paeth", "mixed"]


@dataclass
class CustomResolutionSettings:
    """Settings returned from the CustomResolutionDialog."""

    custom_enabled: bool
    custom_width: int
    custom_height: int
    maintain_ar: bool
    custom_mode: str  # "width", "height", or "2x"
    custom_kernel: str  # "lanczos" or "hermite"
    secondary_enabled: bool
    secondary_mode: str  # "width", "height", or "2x"
    secondary_width: int
    secondary_height: int
    secondary_kernel: str  # "lanczos" or "hermite"
    # Pre-scaling settings
    prescale_enabled: bool
    prescale_mode: str  # "width", "height", or "2x"
    prescale_width: int
    prescale_height: int
    prescale_kernel: str  # "lanczos" or "hermite"


class CustomResolutionDialog(QDialog):
    """
    Dialog for configuring custom resolution and secondary output settings.

    Features:
    - Custom main resolution (downscaled from SR output)
    - Maintain aspect ratio option
    - Secondary resized output settings
    """

    def __init__(
        self,
        parent: QWidget | None = None,
        orig_width: int = 0,
        orig_height: int = 0,
        custom_enabled: bool = False,
        custom_width: int = 0,
        custom_height: int = 0,
        maintain_ar: bool = True,
        custom_mode: str = "width",
        custom_kernel: str = "lanczos",
        secondary_enabled: bool = False,
        secondary_mode: str = "width",
        secondary_width: int = 1920,
        secondary_height: int = 1080,
        secondary_kernel: str = "lanczos",
        prescale_enabled: bool = False,
        prescale_mode: str = "width",
        prescale_width: int = 1920,
        prescale_height: int = 1080,
        prescale_kernel: str = "lanczos",
        upscale_enabled: bool = True,
    ):
        super().__init__(parent)
        self.setWindowTitle("Custom Resolution / Secondary Output / Pre-scale")
        self.resize(500, 520)

        self._orig_width = orig_width
        self._orig_height = orig_height
        self._upscale_enabled = upscale_enabled

        # Custom resolution widgets
        self._custom_check = QCheckBox("Enable")
        self._custom_check.setChecked(custom_enabled)

        self._maintain_ar_check = QCheckBox("Keep aspect ratio")
        self._maintain_ar_check.setChecked(maintain_ar)

        self._custom_mode_combo = QComboBox()
        self._custom_mode_combo.addItems(["Custom width", "Custom height", "2x downscale"])
        if custom_mode == "height":
            self._custom_mode_combo.setCurrentIndex(1)
        elif custom_mode == "2x":
            self._custom_mode_combo.setCurrentIndex(2)
        else:
            self._custom_mode_combo.setCurrentIndex(0)

        self._custom_width_edit = QLineEdit(str(custom_width if custom_width > 0 else "0"))
        self._custom_height_edit = QLineEdit(str(custom_height if custom_height > 0 else "0"))

        self._custom_kernel_combo = QComboBox()
        self._custom_kernel_combo.addItems(["Lanczos", "Hermite"])
        if custom_kernel == "hermite":
            self._custom_kernel_combo.setCurrentIndex(1)
        else:
            self._custom_kernel_combo.setCurrentIndex(0)

        # Secondary output widgets
        self._secondary_check = QCheckBox("Enable")
        self._secondary_check.setChecked(secondary_enabled)

        self._secondary_mode_combo = QComboBox()
        self._secondary_mode_combo.addItems(["Custom width", "Custom height", "2x downscale"])
        if secondary_mode == "height":
            self._secondary_mode_combo.setCurrentIndex(1)
        elif secondary_mode == "2x":
            self._secondary_mode_combo.setCurrentIndex(2)
        else:
            self._secondary_mode_combo.setCurrentIndex(0)

        self._secondary_width_edit = QLineEdit(str(secondary_width))
        self._secondary_height_edit = QLineEdit(str(secondary_height))

        self._secondary_kernel_combo = QComboBox()
        self._secondary_kernel_combo.addItems(["Lanczos", "Hermite"])
        if secondary_kernel == "hermite":
            self._secondary_kernel_combo.setCurrentIndex(1)
        else:
            self._secondary_kernel_combo.setCurrentIndex(0)

        # Pre-scale widgets
        self._prescale_check = QCheckBox("Enable")
        self._prescale_check.setChecked(prescale_enabled)

        self._prescale_mode_combo = QComboBox()
        self._prescale_mode_combo.addItems(["Custom width", "Custom height", "2x downscale"])
        if prescale_mode == "height":
            self._prescale_mode_combo.setCurrentIndex(1)
        elif prescale_mode == "2x":
            self._prescale_mode_combo.setCurrentIndex(2)
        else:
            self._prescale_mode_combo.setCurrentIndex(0)

        self._prescale_width_edit = QLineEdit(str(prescale_width))
        self._prescale_height_edit = QLineEdit(str(prescale_height))

        self._prescale_kernel_combo = QComboBox()
        self._prescale_kernel_combo.addItems(["Lanczos", "Hermite"])
        if prescale_kernel == "hermite":
            self._prescale_kernel_combo.setCurrentIndex(1)
        else:
            self._prescale_kernel_combo.setCurrentIndex(0)

        self._ok_button = QPushButton("OK")
        self._cancel_button = QPushButton("Cancel")

        self._build_ui()
        self._connect_signals()
        self._update_enabled_states()

    def _build_ui(self) -> None:
        """Build the dialog UI."""
        layout = QVBoxLayout()
        self.setLayout(layout)

        # Custom resolution group
        custom_group = QGroupBox("Custom resolution")
        c_layout = QGridLayout()
        custom_group.setLayout(c_layout)

        c_layout.addWidget(self._custom_check, 0, 0, 1, 3)
        c_layout.addWidget(self._maintain_ar_check, 1, 0, 1, 3)
        c_layout.addWidget(QLabel("Mode:"), 2, 0)
        c_layout.addWidget(self._custom_mode_combo, 2, 1, 1, 2)
        c_layout.addWidget(QLabel("Width:"), 3, 0)
        c_layout.addWidget(self._custom_width_edit, 3, 1)
        c_layout.addWidget(QLabel("px"), 3, 2)
        c_layout.addWidget(QLabel("Height:"), 4, 0)
        c_layout.addWidget(self._custom_height_edit, 4, 1)
        c_layout.addWidget(QLabel("px"), 4, 2)
        c_layout.addWidget(QLabel("Kernel:"), 5, 0)
        c_layout.addWidget(self._custom_kernel_combo, 5, 1, 1, 2)

        layout.addWidget(custom_group)

        # Secondary output group
        sec_group = QGroupBox("Secondary output")
        s_layout = QGridLayout()
        sec_group.setLayout(s_layout)

        s_layout.addWidget(self._secondary_check, 0, 0, 1, 3)
        s_layout.addWidget(QLabel("Mode:"), 1, 0)
        s_layout.addWidget(self._secondary_mode_combo, 1, 1, 1, 2)
        s_layout.addWidget(QLabel("Width:"), 2, 0)
        s_layout.addWidget(self._secondary_width_edit, 2, 1)
        s_layout.addWidget(QLabel("px"), 2, 2)
        s_layout.addWidget(QLabel("Height:"), 3, 0)
        s_layout.addWidget(self._secondary_height_edit, 3, 1)
        s_layout.addWidget(QLabel("px"), 3, 2)
        s_layout.addWidget(QLabel("Kernel:"), 4, 0)
        s_layout.addWidget(self._secondary_kernel_combo, 4, 1, 1, 2)

        layout.addWidget(sec_group)

        # Pre-scale group
        self._pre_group = QGroupBox("Pre-scale")
        p_layout = QGridLayout()
        self._pre_group.setLayout(p_layout)

        p_layout.addWidget(self._prescale_check, 0, 0, 1, 3)
        p_layout.addWidget(QLabel("Mode:"), 1, 0)
        p_layout.addWidget(self._prescale_mode_combo, 1, 1, 1, 2)
        p_layout.addWidget(QLabel("Width:"), 2, 0)
        p_layout.addWidget(self._prescale_width_edit, 2, 1)
        p_layout.addWidget(QLabel("px"), 2, 2)
        p_layout.addWidget(QLabel("Height:"), 3, 0)
        p_layout.addWidget(self._prescale_height_edit, 3, 1)
        p_layout.addWidget(QLabel("px"), 3, 2)
        p_layout.addWidget(QLabel("Kernel:"), 4, 0)
        p_layout.addWidget(self._prescale_kernel_combo, 4, 1, 1, 2)

        # Disable pre-scale when upscaling is disabled
        if not self._upscale_enabled:
            self._pre_group.setEnabled(False)
            self._pre_group.setToolTip("Pre-scale is only available when Upscale is enabled")

        layout.addWidget(self._pre_group)

        # Buttons
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()
        btn_layout.addWidget(self._ok_button)
        btn_layout.addWidget(self._cancel_button)
        layout.addLayout(btn_layout)

    def _connect_signals(self) -> None:
        """Connect widget signals."""
        self._custom_check.toggled.connect(self._update_enabled_states)
        self._custom_mode_combo.currentIndexChanged.connect(self._update_enabled_states)
        self._secondary_check.toggled.connect(self._update_enabled_states)
        self._secondary_mode_combo.currentIndexChanged.connect(self._update_enabled_states)
        self._prescale_check.toggled.connect(self._update_enabled_states)
        self._prescale_mode_combo.currentIndexChanged.connect(self._update_enabled_states)
        self._maintain_ar_check.toggled.connect(self._on_maintain_ar_toggled)
        self._custom_width_edit.textChanged.connect(self._on_custom_width_changed)
        self._custom_height_edit.textChanged.connect(self._on_custom_height_changed)
        self._secondary_width_edit.textChanged.connect(self._on_secondary_width_changed)
        self._secondary_height_edit.textChanged.connect(self._on_secondary_height_changed)
        self._prescale_width_edit.textChanged.connect(self._on_prescale_width_changed)
        self._prescale_height_edit.textChanged.connect(self._on_prescale_height_changed)
        self._ok_button.clicked.connect(self.accept)
        self._cancel_button.clicked.connect(self.reject)

    def _update_enabled_states(self) -> None:
        """Update which widgets are enabled based on current settings."""
        custom_on = self._custom_check.isChecked()
        custom_mode_idx = self._custom_mode_combo.currentIndex()
        maintain_ar = self._maintain_ar_check.isChecked()

        # Custom resolution: mode determines which fields are editable
        # Width mode (0): user edits width, height auto-fills if maintain_ar
        # Height mode (1): user edits height, width auto-fills if maintain_ar
        # 2x mode (2): both disabled
        self._custom_mode_combo.setEnabled(custom_on)
        self._maintain_ar_check.setEnabled(custom_on and custom_mode_idx != 2)
        self._custom_kernel_combo.setEnabled(custom_on)

        if custom_mode_idx == 2:  # 2x downscale
            self._custom_width_edit.setEnabled(False)
            self._custom_height_edit.setEnabled(False)
        elif custom_mode_idx == 1:  # height mode
            self._custom_width_edit.setEnabled(custom_on and not maintain_ar)
            self._custom_height_edit.setEnabled(custom_on)
        else:  # width mode
            self._custom_width_edit.setEnabled(custom_on)
            self._custom_height_edit.setEnabled(custom_on and not maintain_ar)

        # Secondary output
        sec_on = self._secondary_check.isChecked()
        mode_idx = self._secondary_mode_combo.currentIndex()

        self._secondary_mode_combo.setEnabled(sec_on)
        self._secondary_kernel_combo.setEnabled(sec_on)
        # Width mode: user edits width, height auto-fills
        # Height mode: user edits height, width auto-fills
        # 2x mode: both disabled
        self._secondary_width_edit.setEnabled(sec_on and mode_idx == 0)
        self._secondary_height_edit.setEnabled(sec_on and mode_idx == 1)

        # Pre-scale
        pre_on = self._prescale_check.isChecked()
        pre_mode_idx = self._prescale_mode_combo.currentIndex()
        self._prescale_mode_combo.setEnabled(pre_on)
        self._prescale_kernel_combo.setEnabled(pre_on)
        self._prescale_width_edit.setEnabled(pre_on and pre_mode_idx == 0)
        self._prescale_height_edit.setEnabled(pre_on and pre_mode_idx == 1)

    def _on_maintain_ar_toggled(self, checked: bool) -> None:
        """Handle maintain aspect ratio checkbox toggle."""
        self._update_enabled_states()
        if checked:
            custom_mode_idx = self._custom_mode_combo.currentIndex()
            if custom_mode_idx == 0:  # width mode
                self._auto_fill_height_from_width()
            elif custom_mode_idx == 1:  # height mode
                self._auto_fill_width_from_height()

    def _on_custom_width_changed(self, text: str) -> None:
        """Auto-fill height when width changes and AR is maintained."""
        custom_mode_idx = self._custom_mode_combo.currentIndex()
        if self._custom_check.isChecked() and self._maintain_ar_check.isChecked():
            if custom_mode_idx == 0:  # width mode
                self._auto_fill_height_from_width()

    def _on_custom_height_changed(self, text: str) -> None:
        """Auto-fill width when height changes and AR is maintained."""
        custom_mode_idx = self._custom_mode_combo.currentIndex()
        if self._custom_check.isChecked() and self._maintain_ar_check.isChecked():
            if custom_mode_idx == 1:  # height mode
                self._auto_fill_width_from_height()

    def _auto_fill_height_from_width(self) -> None:
        """Calculate and set height based on width to maintain aspect ratio."""
        if self._orig_width <= 0 or self._orig_height <= 0:
            return
        try:
            w = int(self._custom_width_edit.text().strip())
        except ValueError:
            return
        if w <= 0:
            return
        h = int(self._orig_height * w / self._orig_width + 0.5)
        self._custom_height_edit.setText(str(h))

    def _auto_fill_width_from_height(self) -> None:
        """Calculate and set width based on height to maintain aspect ratio."""
        if self._orig_width <= 0 or self._orig_height <= 0:
            return
        try:
            h = int(self._custom_height_edit.text().strip())
        except ValueError:
            return
        if h <= 0:
            return
        w = int(self._orig_width * h / self._orig_height + 0.5)
        self._custom_width_edit.setText(str(w))

    def _on_secondary_width_changed(self, text: str) -> None:
        """Auto-fill secondary height when width changes (width mode)."""
        if not self._secondary_check.isChecked():
            return
        if self._secondary_mode_combo.currentIndex() != 0:
            return
        if self._orig_width <= 0 or self._orig_height <= 0:
            return
        try:
            w = int(self._secondary_width_edit.text().strip())
        except ValueError:
            return
        if w <= 0:
            return
        h = int(self._orig_height * w / self._orig_width + 0.5)
        self._secondary_height_edit.setText(str(h))

    def _on_secondary_height_changed(self, text: str) -> None:
        """Auto-fill secondary width when height changes (height mode)."""
        if not self._secondary_check.isChecked():
            return
        if self._secondary_mode_combo.currentIndex() != 1:
            return
        if self._orig_width <= 0 or self._orig_height <= 0:
            return
        try:
            h = int(self._secondary_height_edit.text().strip())
        except ValueError:
            return
        if h <= 0:
            return
        w = int(self._orig_width * h / self._orig_height + 0.5)
        self._secondary_width_edit.setText(str(w))

    def _on_prescale_width_changed(self, text: str) -> None:
        """Auto-fill prescale height when width changes (width mode)."""
        if not self._prescale_check.isChecked():
            return
        if self._prescale_mode_combo.currentIndex() != 0:
            return
        if self._orig_width <= 0 or self._orig_height <= 0:
            return
        try:
            w = int(self._prescale_width_edit.text().strip())
        except ValueError:
            return
        if w <= 0:
            return
        h = int(self._orig_height * w / self._orig_width + 0.5)
        self._prescale_height_edit.setText(str(h))

    def _on_prescale_height_changed(self, text: str) -> None:
        """Auto-fill prescale width when height changes (height mode)."""
        if not self._prescale_check.isChecked():
            return
        if self._prescale_mode_combo.currentIndex() != 1:
            return
        if self._orig_width <= 0 or self._orig_height <= 0:
            return
        try:
            h = int(self._prescale_height_edit.text().strip())
        except ValueError:
            return
        if h <= 0:
            return
        w = int(self._orig_width * h / self._orig_height + 0.5)
        self._prescale_width_edit.setText(str(w))

    def get_settings(self) -> CustomResolutionSettings:
        """
        Get the configured settings.

        Returns:
            CustomResolutionSettings with all dialog values.
        """

        def _to_pos_int(line: QLineEdit, default: int = 0) -> int:
            try:
                v = int(line.text().strip())
                return v if v > 0 else default
            except ValueError:
                return default

        custom_enabled = self._custom_check.isChecked()
        maintain_ar = self._maintain_ar_check.isChecked()

        # Custom mode
        custom_mode_idx = self._custom_mode_combo.currentIndex()
        if custom_mode_idx == 1:
            custom_mode = "height"
        elif custom_mode_idx == 2:
            custom_mode = "2x"
        else:
            custom_mode = "width"

        if custom_enabled:
            cw = _to_pos_int(self._custom_width_edit, 0)
            ch = _to_pos_int(self._custom_height_edit, 0)
        else:
            cw = 0
            ch = 0

        # Custom kernel
        custom_kernel = "hermite" if self._custom_kernel_combo.currentIndex() == 1 else "lanczos"

        # Secondary output
        sec_enabled = self._secondary_check.isChecked()
        mode_idx = self._secondary_mode_combo.currentIndex()

        if mode_idx == 1:
            sec_mode = "height"
        elif mode_idx == 2:
            sec_mode = "2x"
        else:
            sec_mode = "width"

        if sec_enabled and sec_mode == "width":
            sw = _to_pos_int(self._secondary_width_edit, 1920)
            sh = _to_pos_int(self._secondary_height_edit, 1080)
        elif sec_enabled and sec_mode == "height":
            sh = _to_pos_int(self._secondary_height_edit, 1080)
            sw = _to_pos_int(self._secondary_width_edit, 1920)
        else:
            sw = 1920
            sh = 1080

        # Secondary kernel
        secondary_kernel = "hermite" if self._secondary_kernel_combo.currentIndex() == 1 else "lanczos"

        # Pre-scale
        pre_enabled = self._prescale_check.isChecked()
        pre_mode_idx = self._prescale_mode_combo.currentIndex()

        if pre_mode_idx == 1:
            pre_mode = "height"
        elif pre_mode_idx == 2:
            pre_mode = "2x"
        else:
            pre_mode = "width"

        if pre_enabled and pre_mode == "width":
            pw = _to_pos_int(self._prescale_width_edit, 1920)
            ph = _to_pos_int(self._prescale_height_edit, 1080)
        elif pre_enabled and pre_mode == "height":
            ph = _to_pos_int(self._prescale_height_edit, 1080)
            pw = _to_pos_int(self._prescale_width_edit, 1920)
        else:
            pw = 1920
            ph = 1080

        # Pre-scale kernel
        prescale_kernel = "hermite" if self._prescale_kernel_combo.currentIndex() == 1 else "lanczos"

        return CustomResolutionSettings(
            custom_enabled=custom_enabled,
            custom_width=cw,
            custom_height=ch,
            maintain_ar=maintain_ar,
            custom_mode=custom_mode,
            custom_kernel=custom_kernel,
            secondary_enabled=sec_enabled,
            secondary_mode=sec_mode,
            secondary_width=sw,
            secondary_height=sh,
            secondary_kernel=secondary_kernel,
            prescale_enabled=pre_enabled,
            prescale_mode=pre_mode,
            prescale_width=pw,
            prescale_height=ph,
            prescale_kernel=prescale_kernel,
        )


@dataclass
class AnimatedOutputSettings:
    """Settings returned from the AnimatedOutputDialog."""

    output_format: str  # "GIF", "WebP", "AVIF", or "APNG"
    # GIF settings (gifski)
    gif_quality: int  # 1-100 (quality, lower = smaller file)
    gif_fast: bool  # --fast mode (50% faster, 10% worse quality)
    # WebP settings (FFmpeg libwebp)
    webp_quality: int  # 0-100
    webp_lossless: bool
    webp_preset: str  # "none", "default", "picture", etc.
    # AVIF settings (avifenc)
    avif_quality: int  # 0-100 (color quality)
    avif_quality_alpha: int  # 0-100 (alpha quality)
    avif_speed: int  # 0-10 (0=slowest/best, 10=fastest)
    avif_lossless: bool
    # APNG settings (FFmpeg apng encoder)
    apng_pred: str  # "none", "sub", "up", "avg", "paeth", "mixed"

    # For backwards compatibility - generates encoder args string
    @property
    def ffmpeg_args(self) -> str:
        """Generate encoder args string for backwards compatibility."""
        if self.output_format == "GIF":
            args = [f"--quality {self.gif_quality}"]
            if self.gif_fast:
                args.append("--fast")
            return " ".join(args)
        elif self.output_format == "WebP":
            args = []
            if self.webp_lossless:
                args.append("-lossless 1")
            args.append(f"-quality {self.webp_quality}")
            if self.webp_preset != "none":
                args.append(f"-preset {self.webp_preset}")
            return " ".join(args)
        elif self.output_format == "AVIF":
            # Note: AVIF args are handled differently in processor.py
            # This is just for display/logging purposes
            args = [f"-q {self.avif_quality}"]
            args.append(f"--qalpha {self.avif_quality_alpha}")
            args.append(f"-s {self.avif_speed}")
            if self.avif_lossless:
                args.append("-l")
            return " ".join(args)
        elif self.output_format == "APNG":
            return f"-pred {self.apng_pred}"
        return ""


class AnimatedOutputDialog(QDialog):
    """
    Dialog for configuring animated output format and encoder settings.

    Features:
    - Output format selection (GIF, WebP, AVIF, APNG)
    - GIF: Quality, fast mode (gifski)
    - WebP: Quality, lossless mode, preset (FFmpeg libwebp)
    - AVIF: Color quality, alpha quality, speed, lossless mode (avifenc)
    - APNG: Prediction method (FFmpeg apng encoder)
    """

    def __init__(
        self,
        parent: QWidget | None = None,
        output_format: str = "GIF",
        # GIF settings (gifski)
        gif_quality: int = 90,
        gif_fast: bool = False,
        # WebP settings
        webp_quality: int = 90,
        webp_lossless: bool = True,
        webp_preset: str = "none",
        # AVIF settings
        avif_quality: int = 80,
        avif_quality_alpha: int = 90,
        avif_speed: int = 6,
        avif_lossless: bool = False,
        # APNG settings
        apng_pred: str = "mixed",
        # Legacy parameter (ignored, for backwards compatibility)
        ffmpeg_args: str = "",
    ):
        super().__init__(parent)
        self.setWindowTitle("Animated Output Settings")
        self.resize(480, 500)

        # Output format combo
        self._format_combo = QComboBox()
        self._format_combo.addItems(ANIMATED_OUTPUT_FORMATS)
        try:
            idx = ANIMATED_OUTPUT_FORMATS.index(output_format)
            self._format_combo.setCurrentIndex(idx)
        except ValueError:
            self._format_combo.setCurrentIndex(0)

        # === GIF Settings (gifski) ===
        self._gif_quality_slider = QSlider(Qt.Orientation.Horizontal)
        self._gif_quality_slider.setRange(1, 100)
        self._gif_quality_slider.setValue(gif_quality)
        self._gif_quality_slider.setToolTip("Quality: 1 (worst/smallest) to 100 (best/largest)")

        self._gif_quality_spin = QSpinBox()
        self._gif_quality_spin.setRange(1, 100)
        self._gif_quality_spin.setValue(gif_quality)

        self._gif_fast_check = QCheckBox("Fast mode")
        self._gif_fast_check.setChecked(gif_fast)
        self._gif_fast_check.setToolTip("50% faster encoding, but 10% worse quality and larger file")

        # === WebP Settings ===
        self._webp_lossless_check = QCheckBox("Lossless")
        self._webp_lossless_check.setChecked(webp_lossless)
        self._webp_lossless_check.setToolTip("Enable lossless WebP encoding (larger file size)")

        self._webp_quality_slider = QSlider(Qt.Orientation.Horizontal)
        self._webp_quality_slider.setRange(0, 100)
        self._webp_quality_slider.setValue(webp_quality)
        self._webp_quality_slider.setToolTip("Quality: 0 (worst) to 100 (best)")

        self._webp_quality_spin = QSpinBox()
        self._webp_quality_spin.setRange(0, 100)
        self._webp_quality_spin.setValue(webp_quality)

        self._webp_preset_combo = QComboBox()
        self._webp_preset_combo.addItems(WEBP_PRESETS)
        try:
            preset_idx = WEBP_PRESETS.index(webp_preset)
            self._webp_preset_combo.setCurrentIndex(preset_idx)
        except ValueError:
            self._webp_preset_combo.setCurrentIndex(0)
        self._webp_preset_combo.setToolTip(
            "Preset for WebP encoding:\n"
            "- none: No preset (default)\n"
            "- picture: Digital pictures (portraits, indoor)\n"
            "- photo: Outdoor photographs\n"
            "- drawing: Drawings with high-contrast details\n"
            "- icon: Small colorful images\n"
            "- text: Text-like images"
        )

        # === AVIF Settings ===
        self._avif_lossless_check = QCheckBox("Lossless")
        self._avif_lossless_check.setChecked(avif_lossless)
        self._avif_lossless_check.setToolTip("Enable lossless AVIF encoding (larger file size)")

        self._avif_quality_slider = QSlider(Qt.Orientation.Horizontal)
        self._avif_quality_slider.setRange(0, 100)
        self._avif_quality_slider.setValue(avif_quality)
        self._avif_quality_slider.setToolTip("Color quality: 0 (worst) to 100 (lossless)")

        self._avif_quality_spin = QSpinBox()
        self._avif_quality_spin.setRange(0, 100)
        self._avif_quality_spin.setValue(avif_quality)

        self._avif_alpha_slider = QSlider(Qt.Orientation.Horizontal)
        self._avif_alpha_slider.setRange(0, 100)
        self._avif_alpha_slider.setValue(avif_quality_alpha)
        self._avif_alpha_slider.setToolTip("Alpha quality: 0 (worst) to 100 (lossless)")

        self._avif_alpha_spin = QSpinBox()
        self._avif_alpha_spin.setRange(0, 100)
        self._avif_alpha_spin.setValue(avif_quality_alpha)

        self._avif_speed_slider = QSlider(Qt.Orientation.Horizontal)
        self._avif_speed_slider.setRange(0, 10)
        self._avif_speed_slider.setValue(avif_speed)
        self._avif_speed_slider.setToolTip("Speed: 0 (slowest/best) to 10 (fastest)")

        self._avif_speed_spin = QSpinBox()
        self._avif_speed_spin.setRange(0, 10)
        self._avif_speed_spin.setValue(avif_speed)

        # === APNG Settings ===
        self._apng_pred_combo = QComboBox()
        self._apng_pred_combo.addItems(APNG_PRED_METHODS)
        try:
            pred_idx = APNG_PRED_METHODS.index(apng_pred)
            self._apng_pred_combo.setCurrentIndex(pred_idx)
        except ValueError:
            self._apng_pred_combo.setCurrentIndex(5)  # "mixed" default
        self._apng_pred_combo.setToolTip(
            "Prediction method for PNG compression:\n"
            "- none: No prediction (fastest, largest files)\n"
            "- sub: Predict from left pixel\n"
            "- up: Predict from above pixel\n"
            "- avg: Average of left and above\n"
            "- paeth: Paeth predictor (good compression)\n"
            "- mixed: Try all filters per line (slowest, best compression)"
        )

        # Buttons
        self._ok_button = QPushButton("OK")
        self._cancel_button = QPushButton("Cancel")
        self._reset_button = QPushButton("Reset to Defaults")

        self._build_ui()
        self._connect_signals()
        self._update_format_visibility()

    def _build_ui(self) -> None:
        """Build the dialog UI."""
        layout = QVBoxLayout()
        self.setLayout(layout)

        # Format selection group
        format_group = QGroupBox("Output Format")
        format_layout = QHBoxLayout()
        format_group.setLayout(format_layout)
        format_layout.addWidget(QLabel("Format:"))
        format_layout.addWidget(self._format_combo)
        format_layout.addStretch()
        layout.addWidget(format_group)

        # GIF settings group (gifski)
        self._gif_group = QGroupBox("GIF Settings (gifski)")
        gif_layout = QGridLayout()
        self._gif_group.setLayout(gif_layout)

        gif_layout.addWidget(QLabel("Quality:"), 0, 0)
        gif_layout.addWidget(self._gif_quality_slider, 0, 1)
        gif_layout.addWidget(self._gif_quality_spin, 0, 2)

        gif_layout.addWidget(self._gif_fast_check, 1, 0, 1, 3)

        gif_hint = QLabel("(gifski provides superior GIF quality with temporal dithering)")
        gif_hint.setStyleSheet("color: gray; font-size: 10px;")
        gif_layout.addWidget(gif_hint, 2, 0, 1, 3)

        layout.addWidget(self._gif_group)

        # WebP settings group
        self._webp_group = QGroupBox("WebP Settings (FFmpeg)")
        webp_layout = QGridLayout()
        self._webp_group.setLayout(webp_layout)

        webp_layout.addWidget(self._webp_lossless_check, 0, 0, 1, 3)

        webp_layout.addWidget(QLabel("Quality:"), 1, 0)
        webp_layout.addWidget(self._webp_quality_slider, 1, 1)
        webp_layout.addWidget(self._webp_quality_spin, 1, 2)

        webp_layout.addWidget(QLabel("Preset:"), 2, 0)
        webp_layout.addWidget(self._webp_preset_combo, 2, 1, 1, 2)

        layout.addWidget(self._webp_group)

        # AVIF settings group
        self._avif_group = QGroupBox("AVIF Settings (avifenc)")
        avif_layout = QGridLayout()
        self._avif_group.setLayout(avif_layout)

        avif_layout.addWidget(self._avif_lossless_check, 0, 0, 1, 3)

        avif_layout.addWidget(QLabel("Color Quality:"), 1, 0)
        avif_layout.addWidget(self._avif_quality_slider, 1, 1)
        avif_layout.addWidget(self._avif_quality_spin, 1, 2)

        avif_layout.addWidget(QLabel("Alpha Quality:"), 2, 0)
        avif_layout.addWidget(self._avif_alpha_slider, 2, 1)
        avif_layout.addWidget(self._avif_alpha_spin, 2, 2)

        avif_layout.addWidget(QLabel("Speed:"), 3, 0)
        avif_layout.addWidget(self._avif_speed_slider, 3, 1)
        avif_layout.addWidget(self._avif_speed_spin, 3, 2)

        speed_hint = QLabel("(0 = slowest/best quality, 10 = fastest)")
        speed_hint.setStyleSheet("color: gray; font-size: 10px;")
        avif_layout.addWidget(speed_hint, 4, 1, 1, 2)

        layout.addWidget(self._avif_group)

        # APNG settings group
        self._apng_group = QGroupBox("APNG Settings (FFmpeg)")
        apng_layout = QGridLayout()
        self._apng_group.setLayout(apng_layout)

        apng_layout.addWidget(QLabel("Prediction:"), 0, 0)
        apng_layout.addWidget(self._apng_pred_combo, 0, 1)

        pred_hint = QLabel("(none = fastest, mixed = best compression)")
        pred_hint.setStyleSheet("color: gray; font-size: 10px;")
        apng_layout.addWidget(pred_hint, 1, 0, 1, 2)

        layout.addWidget(self._apng_group)

        # Info label
        info_label = QLabel(
            "Note: These settings apply to animated GIF/video inputs.\n"
            "GIF requires gifski. WebP/APNG require FFmpeg.\n"
            "AVIF requires avifenc (libavif)."
        )
        info_label.setWordWrap(True)
        layout.addWidget(info_label)

        # Buttons
        btn_layout = QHBoxLayout()
        btn_layout.addWidget(self._reset_button)
        btn_layout.addStretch()
        btn_layout.addWidget(self._ok_button)
        btn_layout.addWidget(self._cancel_button)
        layout.addLayout(btn_layout)

    def _connect_signals(self) -> None:
        """Connect widget signals."""
        self._format_combo.currentTextChanged.connect(self._update_format_visibility)

        # GIF slider/spin sync
        self._gif_quality_slider.valueChanged.connect(self._gif_quality_spin.setValue)
        self._gif_quality_spin.valueChanged.connect(self._gif_quality_slider.setValue)

        # WebP slider/spin sync
        self._webp_quality_slider.valueChanged.connect(self._webp_quality_spin.setValue)
        self._webp_quality_spin.valueChanged.connect(self._webp_quality_slider.setValue)
        self._webp_lossless_check.toggled.connect(self._on_webp_lossless_toggled)

        # AVIF slider/spin sync
        self._avif_quality_slider.valueChanged.connect(self._avif_quality_spin.setValue)
        self._avif_quality_spin.valueChanged.connect(self._avif_quality_slider.setValue)
        self._avif_alpha_slider.valueChanged.connect(self._avif_alpha_spin.setValue)
        self._avif_alpha_spin.valueChanged.connect(self._avif_alpha_slider.setValue)
        self._avif_speed_slider.valueChanged.connect(self._avif_speed_spin.setValue)
        self._avif_speed_spin.valueChanged.connect(self._avif_speed_slider.setValue)
        self._avif_lossless_check.toggled.connect(self._on_avif_lossless_toggled)

        self._reset_button.clicked.connect(self._reset_to_defaults)
        self._ok_button.clicked.connect(self.accept)
        self._cancel_button.clicked.connect(self.reject)

    def _update_format_visibility(self) -> None:
        """Show/hide settings groups based on selected format."""
        fmt = self._format_combo.currentText()
        self._gif_group.setVisible(fmt == "GIF")
        self._webp_group.setVisible(fmt == "WebP")
        self._avif_group.setVisible(fmt == "AVIF")
        self._apng_group.setVisible(fmt == "APNG")

    def _on_webp_lossless_toggled(self, checked: bool) -> None:
        """Disable quality slider when lossless is enabled."""
        self._webp_quality_slider.setEnabled(not checked)
        self._webp_quality_spin.setEnabled(not checked)

    def _on_avif_lossless_toggled(self, checked: bool) -> None:
        """Disable quality sliders when lossless is enabled."""
        self._avif_quality_slider.setEnabled(not checked)
        self._avif_quality_spin.setEnabled(not checked)
        self._avif_alpha_slider.setEnabled(not checked)
        self._avif_alpha_spin.setEnabled(not checked)

    def _reset_to_defaults(self) -> None:
        """Reset all settings to defaults."""
        # GIF defaults
        self._gif_quality_slider.setValue(90)
        self._gif_fast_check.setChecked(False)

        # WebP defaults
        self._webp_quality_slider.setValue(90)
        self._webp_lossless_check.setChecked(True)
        self._webp_preset_combo.setCurrentIndex(0)  # "none"

        # AVIF defaults
        self._avif_quality_slider.setValue(80)
        self._avif_alpha_slider.setValue(90)
        self._avif_speed_slider.setValue(6)
        self._avif_lossless_check.setChecked(False)

        # APNG defaults
        self._apng_pred_combo.setCurrentIndex(5)  # "mixed"

    def get_settings(self) -> AnimatedOutputSettings:
        """
        Get the configured settings.

        Returns:
            AnimatedOutputSettings with dialog values.
        """
        return AnimatedOutputSettings(
            output_format=self._format_combo.currentText(),
            gif_quality=self._gif_quality_spin.value(),
            gif_fast=self._gif_fast_check.isChecked(),
            webp_quality=self._webp_quality_spin.value(),
            webp_lossless=self._webp_lossless_check.isChecked(),
            webp_preset=self._webp_preset_combo.currentText(),
            avif_quality=self._avif_quality_spin.value(),
            avif_quality_alpha=self._avif_alpha_spin.value(),
            avif_speed=self._avif_speed_spin.value(),
            avif_lossless=self._avif_lossless_check.isChecked(),
            apng_pred=self._apng_pred_combo.currentText(),
        )
