"""
Dialog windows for the GUI.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

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
)

if TYPE_CHECKING:
    from PySide6.QtWidgets import QWidget


@dataclass
class CustomResolutionSettings:
    """Settings returned from the CustomResolutionDialog."""

    custom_enabled: bool
    custom_width: int
    custom_height: int
    maintain_ar: bool
    secondary_enabled: bool
    secondary_mode: str  # "width", "height", or "2x"
    secondary_width: int
    secondary_height: int
    # Pre-scaling settings
    prescale_enabled: bool
    prescale_mode: str  # "width", "height", or "2x"
    prescale_width: int
    prescale_height: int
    # Kernel selection
    kernel: str  # "lanczos" or "hermite"


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
        secondary_enabled: bool = False,
        secondary_mode: str = "width",
        secondary_width: int = 1920,
        secondary_height: int = 1080,
        prescale_enabled: bool = False,
        prescale_mode: str = "width",
        prescale_width: int = 1920,
        prescale_height: int = 1080,
        kernel: str = "lanczos",
    ):
        super().__init__(parent)
        self.setWindowTitle("Custom Resolution / Secondary Output / Pre-scaling")
        self.resize(500, 480)

        self._orig_width = orig_width
        self._orig_height = orig_height

        # Widgets
        self._custom_check = QCheckBox("Enable custom main resolution")
        self._custom_check.setChecked(custom_enabled)

        self._maintain_ar_check = QCheckBox("Maintain aspect ratio (main)")
        self._maintain_ar_check.setChecked(maintain_ar)

        self._custom_width_edit = QLineEdit(str(custom_width if custom_width > 0 else "0"))
        self._custom_height_edit = QLineEdit(str(custom_height if custom_height > 0 else "0"))

        # Secondary
        self._secondary_check = QCheckBox("Enable secondary resized output")
        self._secondary_check.setChecked(secondary_enabled)

        self._secondary_mode_combo = QComboBox()
        self._secondary_mode_combo.addItems(["Custom width", "Custom height", "2x from main"])
        if secondary_mode == "height":
            self._secondary_mode_combo.setCurrentIndex(1)
        elif secondary_mode == "2x":
            self._secondary_mode_combo.setCurrentIndex(2)
        else:
            self._secondary_mode_combo.setCurrentIndex(0)

        self._secondary_width_edit = QLineEdit(str(secondary_width))
        self._secondary_height_edit = QLineEdit(str(secondary_height))

        # Pre-scaling widgets
        self._prescale_check = QCheckBox("Enable pre-scaling (downscale before upscaling)")
        self._prescale_check.setChecked(prescale_enabled)

        self._prescale_mode_combo = QComboBox()
        self._prescale_mode_combo.addItems(["Custom width", "Custom height", "2x (half size)"])
        if prescale_mode == "height":
            self._prescale_mode_combo.setCurrentIndex(1)
        elif prescale_mode == "2x":
            self._prescale_mode_combo.setCurrentIndex(2)
        else:
            self._prescale_mode_combo.setCurrentIndex(0)

        self._prescale_width_edit = QLineEdit(str(prescale_width))
        self._prescale_height_edit = QLineEdit(str(prescale_height))

        # Kernel selection
        self._kernel_combo = QComboBox()
        self._kernel_combo.addItems(["Lanczos", "Hermite"])
        if kernel == "hermite":
            self._kernel_combo.setCurrentIndex(1)
        else:
            self._kernel_combo.setCurrentIndex(0)

        self._ok_button = QPushButton("OK")
        self._cancel_button = QPushButton("Cancel")

        self._build_ui()
        self._connect_signals()
        self._update_enabled_states()

    def _build_ui(self) -> None:
        """Build the dialog UI."""
        layout = QVBoxLayout()
        self.setLayout(layout)

        # Custom main resolution group
        custom_group = QGroupBox("Custom main resolution (from SR output)")
        c_layout = QGridLayout()
        custom_group.setLayout(c_layout)

        c_layout.addWidget(self._custom_check, 0, 0, 1, 3)
        c_layout.addWidget(self._maintain_ar_check, 1, 0, 1, 3)
        c_layout.addWidget(QLabel("Width:"), 2, 0)
        c_layout.addWidget(self._custom_width_edit, 2, 1)
        c_layout.addWidget(QLabel("px"), 2, 2)
        c_layout.addWidget(QLabel("Height:"), 3, 0)
        c_layout.addWidget(self._custom_height_edit, 3, 1)
        c_layout.addWidget(QLabel("px"), 3, 2)

        layout.addWidget(custom_group)

        # Secondary output group
        sec_group = QGroupBox("Secondary resized output")
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
        s_layout.addWidget(QLabel("(Aspect ratio preserved in width/height modes)"), 4, 0, 1, 3)

        layout.addWidget(sec_group)

        # Pre-scaling group
        pre_group = QGroupBox("Pre-scaling (downscale input before upscaling)")
        p_layout = QGridLayout()
        pre_group.setLayout(p_layout)

        p_layout.addWidget(self._prescale_check, 0, 0, 1, 3)
        p_layout.addWidget(QLabel("Mode:"), 1, 0)
        p_layout.addWidget(self._prescale_mode_combo, 1, 1, 1, 2)
        p_layout.addWidget(QLabel("Width:"), 2, 0)
        p_layout.addWidget(self._prescale_width_edit, 2, 1)
        p_layout.addWidget(QLabel("px"), 2, 2)
        p_layout.addWidget(QLabel("Height:"), 3, 0)
        p_layout.addWidget(self._prescale_height_edit, 3, 1)
        p_layout.addWidget(QLabel("px"), 3, 2)
        p_layout.addWidget(QLabel("(Aspect ratio preserved in width/height modes)"), 4, 0, 1, 3)

        layout.addWidget(pre_group)

        # Kernel selection group
        kernel_group = QGroupBox("Scaling kernel")
        k_layout = QHBoxLayout()
        kernel_group.setLayout(k_layout)
        k_layout.addWidget(QLabel("Kernel:"))
        k_layout.addWidget(self._kernel_combo)
        k_layout.addStretch()

        layout.addWidget(kernel_group)

        # Buttons
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()
        btn_layout.addWidget(self._ok_button)
        btn_layout.addWidget(self._cancel_button)
        layout.addLayout(btn_layout)

    def _connect_signals(self) -> None:
        """Connect widget signals."""
        self._custom_check.toggled.connect(self._update_enabled_states)
        self._secondary_check.toggled.connect(self._update_enabled_states)
        self._secondary_mode_combo.currentIndexChanged.connect(self._update_enabled_states)
        self._prescale_check.toggled.connect(self._update_enabled_states)
        self._prescale_mode_combo.currentIndexChanged.connect(self._update_enabled_states)
        self._maintain_ar_check.toggled.connect(self._on_maintain_ar_toggled)
        self._custom_width_edit.textChanged.connect(self._on_custom_width_changed)
        self._secondary_width_edit.textChanged.connect(self._on_secondary_width_changed)
        self._secondary_height_edit.textChanged.connect(self._on_secondary_height_changed)
        self._prescale_width_edit.textChanged.connect(self._on_prescale_width_changed)
        self._prescale_height_edit.textChanged.connect(self._on_prescale_height_changed)
        self._ok_button.clicked.connect(self.accept)
        self._cancel_button.clicked.connect(self.reject)

    def _update_enabled_states(self) -> None:
        """Update which widgets are enabled based on current settings."""
        custom_on = self._custom_check.isChecked()
        self._custom_width_edit.setEnabled(custom_on)
        self._custom_height_edit.setEnabled(custom_on and not self._maintain_ar_check.isChecked())

        sec_on = self._secondary_check.isChecked()
        mode_idx = self._secondary_mode_combo.currentIndex()

        # Width mode: user edits width, height auto-fills
        # Height mode: user edits height, width auto-fills
        # 2x mode: both disabled
        self._secondary_width_edit.setEnabled(sec_on and mode_idx == 0)
        self._secondary_height_edit.setEnabled(sec_on and mode_idx == 1)

        # Pre-scaling
        pre_on = self._prescale_check.isChecked()
        pre_mode_idx = self._prescale_mode_combo.currentIndex()
        self._prescale_width_edit.setEnabled(pre_on and pre_mode_idx == 0)
        self._prescale_height_edit.setEnabled(pre_on and pre_mode_idx == 1)

    def _on_maintain_ar_toggled(self, checked: bool) -> None:
        """Handle maintain aspect ratio checkbox toggle."""
        custom_on = self._custom_check.isChecked()
        self._custom_height_edit.setEnabled(custom_on and not checked)
        if checked:
            self._auto_fill_height_main()

    def _on_custom_width_changed(self, text: str) -> None:
        """Auto-fill height when width changes and AR is maintained."""
        if self._custom_check.isChecked() and self._maintain_ar_check.isChecked():
            self._auto_fill_height_main()

    def _auto_fill_height_main(self) -> None:
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

        if custom_enabled:
            cw = _to_pos_int(self._custom_width_edit, 0)
            ch = _to_pos_int(self._custom_height_edit, 0)
        else:
            cw = 0
            ch = 0

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

        # Pre-scaling
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

        # Kernel
        kernel = "hermite" if self._kernel_combo.currentIndex() == 1 else "lanczos"

        return CustomResolutionSettings(
            custom_enabled=custom_enabled,
            custom_width=cw,
            custom_height=ch,
            maintain_ar=maintain_ar,
            secondary_enabled=sec_enabled,
            secondary_mode=sec_mode,
            secondary_width=sw,
            secondary_height=sh,
            prescale_enabled=pre_enabled,
            prescale_mode=pre_mode,
            prescale_width=pw,
            prescale_height=ph,
            kernel=kernel,
        )
