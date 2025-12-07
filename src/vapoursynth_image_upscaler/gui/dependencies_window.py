"""
Dependencies installation window for VapourSynth Image Upscaler.

Handles installation of:
- Python packages via pip (PySide6, vsjetpack, Pillow, numpy)
- VapourSynth plugins (extracted to %APPDATA%/Vapoursynth/plugins64)
- External tools (ffmpeg, gifski, avifenc, pngquant, pingo) with PATH setup

Note: VapourSynth itself must be installed manually by the user.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import TYPE_CHECKING
from urllib.request import Request, urlopen

from PySide6.QtCore import Qt, QThread, Signal, QTimer
from PySide6.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QProgressBar,
    QTextEdit,
    QGroupBox,
    QMessageBox,
    QScrollArea,
    QWidget,
    QFrame,
)

if TYPE_CHECKING:
    from PySide6.QtWidgets import QWidget as QWidgetType


# === Configuration ===

# Python packages to install via pip
# Note: VapourSynth is NOT included - it must be installed manually by the user
PIP_PACKAGES = ["PySide6", "vsjetpack", "Pillow", "numpy"]

# VapourSynth plugins to download
# Format: (name, url, extract_type, special_handling)
# extract_type: "zip", "7z", "7z_multi" (for multi-part archives)
# special_handling: None, "copy_fmtconv", "copy_imwri"
VS_PLUGINS = [
    (
        "akarin-vs",
        "https://github.com/Jaded-Encoding-Thaumaturgy/akarin-vapoursynth-plugin/releases/download/v1.1.0/akarin-win64-release.zip",
        "zip",
        None,
    ),
    (
        "fmtconv",
        "https://github.com/EleonoreMizo/fmtconv/releases/download/r30/fmtconv-r30.zip",
        "zip",
        "copy_fmtconv",  # Copy fmtconv.dll from win64 folder
    ),
    (
        "resize2",
        "https://github.com/Jaded-Encoding-Thaumaturgy/vapoursynth-resize2/releases/download/0.3.3/resize2-win64.zip",
        "zip",
        None,
    ),
    (
        "vapoursynth-zip",
        "https://github.com/dnjulek/vapoursynth-zip/releases/download/R10/vapoursynth-zip-r7-windows-x86_64.zip",
        "zip",
        None,
    ),
    (
        "zsmooth",
        "https://github.com/adworacz/zsmooth/releases/download/0.15/zsmooth-x86_64-windows.zip",
        "zip",
        None,
    ),
    (
        "bestsource",
        "https://github.com/vapoursynth/bestsource/releases/download/R15/BestSource-R15.7z",
        "7z",
        None,
    ),
    (
        "vs-imwri",
        "https://github.com/vapoursynth/vs-imwri/releases/download/R2/imwri-r2.7z",
        "7z",
        "copy_imwri",  # Copy libimwri.dll from win64 folder
    ),
    (
        "deblock",
        "https://github.com/HomeOfVapourSynthEvolution/VapourSynth-Deblock/releases/download/r7.1/Deblock-r7.1.7z",
        "7z",
        None,
    ),
    (
        "dctfilter",
        "https://github.com/AmusementClub/VapourSynth-DCTFilter/releases/download/r3.1A/release-x64.zip",
        "zip",
        None,
    ),
    (
        "vsmlrt (part 1/2)",
        "https://github.com/AmusementClub/vs-mlrt/releases/download/v15.14/vsmlrt-windows-x64-tensorrt.v15.14.7z.001",
        "7z_multi_1",
        None,
    ),
    (
        "vsmlrt (part 2/2)",
        "https://github.com/AmusementClub/vs-mlrt/releases/download/v15.14/vsmlrt-windows-x64-tensorrt.v15.14.7z.002",
        "7z_multi_2",
        None,
    ),
]

# External tools to download
# Format: (name, url, extract_type, path_subdir)
# path_subdir: subdirectory within TOOLS_DIR to add to PATH
EXTERNAL_TOOLS = [
    (
        "ffmpeg",
        "https://github.com/nekotrix/FFmpeg-Builds-SVT-AV1-Essential/releases/download/latest/ffmpeg-master-latest-win64-gpl.zip",
        "zip",
        "ffmpeg-master-latest-win64-gpl/bin",  # Add this subdir to PATH
    ),
    (
        "gifski",
        "https://gif.ski/gifski-1.32.0.zip",
        "zip",
        "win",  # Add this subdir to PATH
    ),
    (
        "avifenc",
        "https://github.com/AOMediaCodec/libavif/releases/download/v1.3.0/windows-artifacts.zip",
        "zip",
        "",  # Add TOOLS_DIR directly to PATH
    ),
    (
        "pngquant",
        "https://github.com/jibsen/pngquant-winbuild/releases/download/v2.17.0/pngquant-2.17.0-win-x64.zip",
        "zip",
        "",  # Add TOOLS_DIR directly to PATH
    ),
    (
        "pingo",
        "https://css-ig.net/bin/pingo-win64.zip",
        "zip",
        "",  # Add TOOLS_DIR directly to PATH
    ),
]

# Installation directories
APPDATA = Path(os.environ.get("APPDATA", ""))
VS_PLUGINS_DIR = APPDATA / "Vapoursynth" / "plugins64"
TOOLS_DIR = APPDATA / "vapoursynth-image-upscaler-GUI"


class InstallWorker(QThread):
    """Worker thread for installing dependencies."""

    progress_signal = Signal(str)  # Log message
    status_signal = Signal(str, int, int)  # (status, current, total)
    finished_signal = Signal(bool, str)  # (success, message)

    def __init__(
        self,
        install_pip: bool = True,
        install_plugins: bool = True,
        install_tools: bool = True,
    ):
        super().__init__()
        self._install_pip = install_pip
        self._install_plugins = install_plugins
        self._install_tools = install_tools
        self._cancelled = False

    def cancel(self) -> None:
        """Request cancellation."""
        self._cancelled = True

    def run(self) -> None:
        """Run the installation process."""
        try:
            total_steps = 0
            if self._install_pip:
                total_steps += len(PIP_PACKAGES)
            if self._install_plugins:
                total_steps += len(VS_PLUGINS)
            if self._install_tools:
                total_steps += len(EXTERNAL_TOOLS)

            current_step = 0

            # Install pip packages
            if self._install_pip:
                self.progress_signal.emit("\n=== Installing Python packages ===\n")
                for pkg in PIP_PACKAGES:
                    if self._cancelled:
                        self.finished_signal.emit(False, "Installation cancelled.")
                        return
                    current_step += 1
                    self.status_signal.emit(f"Installing {pkg}...", current_step, total_steps)
                    self.progress_signal.emit(f"Installing {pkg}...")
                    success, msg = self._install_pip_package(pkg)
                    self.progress_signal.emit(msg)
                    if not success:
                        self.progress_signal.emit(f"Warning: Failed to install {pkg}")

            # Install VS plugins
            if self._install_plugins:
                self.progress_signal.emit("\n=== Installing VapourSynth plugins ===\n")
                # Ensure plugins directory exists
                VS_PLUGINS_DIR.mkdir(parents=True, exist_ok=True)

                # Handle multi-part archives specially
                vsmlrt_parts: list[Path] = []

                for name, url, extract_type, special in VS_PLUGINS:
                    if self._cancelled:
                        self.finished_signal.emit(False, "Installation cancelled.")
                        return
                    current_step += 1
                    self.status_signal.emit(f"Downloading {name}...", current_step, total_steps)
                    self.progress_signal.emit(f"Downloading {name}...")

                    if extract_type.startswith("7z_multi"):
                        # Download multi-part archive
                        temp_file = self._download_file(url, name)
                        if temp_file:
                            vsmlrt_parts.append(temp_file)
                            self.progress_signal.emit(f"  Downloaded part to: {temp_file}")
                            # Extract when we have all parts
                            if extract_type == "7z_multi_2" and len(vsmlrt_parts) == 2:
                                self.progress_signal.emit("Extracting vsmlrt (multi-part archive)...")
                                success = self._extract_7z_multi(vsmlrt_parts)
                                if success:
                                    self.progress_signal.emit("  vsmlrt extracted successfully")
                                else:
                                    self.progress_signal.emit("  Warning: Failed to extract vsmlrt")
                                # Clean up
                                for p in vsmlrt_parts:
                                    try:
                                        p.unlink()
                                    except Exception:
                                        pass
                    else:
                        success = self._download_and_extract_plugin(name, url, extract_type, special)
                        if success:
                            self.progress_signal.emit(f"  {name} installed successfully")
                        else:
                            self.progress_signal.emit(f"  Warning: Failed to install {name}")

            # Install external tools
            if self._install_tools:
                self.progress_signal.emit("\n=== Installing external tools ===\n")
                TOOLS_DIR.mkdir(parents=True, exist_ok=True)

                paths_to_add: list[Path] = []

                for name, url, extract_type, path_subdir in EXTERNAL_TOOLS:
                    if self._cancelled:
                        self.finished_signal.emit(False, "Installation cancelled.")
                        return
                    current_step += 1
                    self.status_signal.emit(f"Downloading {name}...", current_step, total_steps)
                    self.progress_signal.emit(f"Downloading {name}...")
                    success = self._download_and_extract_tool(name, url, extract_type)
                    if success:
                        self.progress_signal.emit(f"  {name} installed successfully")
                        # Track the path to add
                        if path_subdir:
                            paths_to_add.append(TOOLS_DIR / path_subdir)
                        else:
                            paths_to_add.append(TOOLS_DIR)
                    else:
                        self.progress_signal.emit(f"  Warning: Failed to install {name}")

                # Add all tool directories to PATH
                self.progress_signal.emit("\nAdding tools directories to PATH...")
                for path_dir in paths_to_add:
                    if path_dir.exists():
                        success = self._add_to_path(path_dir)
                        if success:
                            self.progress_signal.emit(f"  Added {path_dir} to user PATH")
                        else:
                            self.progress_signal.emit(f"  Warning: Could not add {path_dir} to PATH")
                    else:
                        self.progress_signal.emit(f"  Warning: Path does not exist: {path_dir}")

            self.progress_signal.emit("\n=== Installation complete ===\n")
            self.finished_signal.emit(True, "All dependencies installed successfully!")

        except Exception as e:
            self.finished_signal.emit(False, f"Installation failed: {e}")

    def _install_pip_package(self, package: str) -> tuple[bool, str]:
        """Install a Python package via pip."""
        try:
            result = subprocess.run(
                [sys.executable, "-m", "pip", "install", "--upgrade", package],
                capture_output=True,
                text=True,
                timeout=300,
            )
            if result.returncode == 0:
                return True, f"  {package} installed successfully"
            else:
                return False, f"  Error: {result.stderr}"
        except subprocess.TimeoutExpired:
            return False, f"  Timeout installing {package}"
        except Exception as e:
            return False, f"  Error: {e}"

    def _download_file(self, url: str, name: str) -> Path | None:
        """Download a file to temp directory."""
        try:
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
            }
            req = Request(url, headers=headers)

            # Get filename from URL
            filename = url.split("/")[-1]
            temp_path = Path(tempfile.gettempdir()) / f"vsiu_dep_{filename}"

            with urlopen(req, timeout=120) as response:
                with open(temp_path, "wb") as f:
                    # Download in chunks
                    while True:
                        chunk = response.read(8192)
                        if not chunk:
                            break
                        f.write(chunk)

            return temp_path
        except Exception as e:
            self.progress_signal.emit(f"  Download error: {e}")
            return None

    def _download_and_extract_plugin(
        self, name: str, url: str, extract_type: str, special: str | None
    ) -> bool:
        """Download and extract a VapourSynth plugin."""
        # Handle direct DLL download (no extraction needed)
        if extract_type == "dll":
            return self._download_dll_plugin(url, name)

        temp_file = self._download_file(url, name)
        if not temp_file:
            return False

        try:
            if extract_type == "zip":
                return self._extract_zip_plugin(temp_file, special)
            elif extract_type == "7z":
                return self._extract_7z_plugin(temp_file, special)
            else:
                return False
        finally:
            try:
                temp_file.unlink()
            except Exception:
                pass

    def _download_dll_plugin(self, url: str, name: str) -> bool:
        """Download a DLL directly to VS plugins directory."""
        try:
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
            }
            req = Request(url, headers=headers)

            # Get filename from URL
            filename = url.split("/")[-1]
            dest_path = VS_PLUGINS_DIR / filename

            with urlopen(req, timeout=120) as response:
                with open(dest_path, "wb") as f:
                    while True:
                        chunk = response.read(8192)
                        if not chunk:
                            break
                        f.write(chunk)

            return True
        except Exception as e:
            self.progress_signal.emit(f"  Download error: {e}")
            return False

    def _extract_zip_plugin(self, zip_path: Path, special: str | None) -> bool:
        """Extract a zip plugin to VS plugins directory."""
        try:
            with zipfile.ZipFile(zip_path, "r") as zf:
                # Create temp extraction dir
                with tempfile.TemporaryDirectory() as temp_dir:
                    zf.extractall(temp_dir)
                    temp_path = Path(temp_dir)

                    if special == "copy_fmtconv":
                        # Find and copy fmtconv.dll (usually in win64 subfolder)
                        for dll in temp_path.rglob("**/win64/fmtconv.dll"):
                            shutil.copy2(dll, VS_PLUGINS_DIR / "fmtconv.dll")
                            return True
                        # Fallback: search anywhere
                        for dll in temp_path.rglob("fmtconv.dll"):
                            shutil.copy2(dll, VS_PLUGINS_DIR / "fmtconv.dll")
                            return True
                        return False
                    elif special == "copy_imwri":
                        # Find and copy libimwri.dll from win64 folder
                        for dll in temp_path.rglob("**/win64/libimwri.dll"):
                            shutil.copy2(dll, VS_PLUGINS_DIR / "libimwri.dll")
                            return True
                        # Fallback: search anywhere for libimwri.dll
                        for dll in temp_path.rglob("libimwri.dll"):
                            shutil.copy2(dll, VS_PLUGINS_DIR / "libimwri.dll")
                            return True
                        # Try imwri.dll as last resort
                        for dll in temp_path.rglob("imwri.dll"):
                            shutil.copy2(dll, VS_PLUGINS_DIR / "imwri.dll")
                            return True
                        return False
                    else:
                        # Copy all DLL files
                        for dll in temp_path.rglob("*.dll"):
                            shutil.copy2(dll, VS_PLUGINS_DIR / dll.name)
                        return True
        except Exception as e:
            self.progress_signal.emit(f"  Extract error: {e}")
            return False

    def _extract_7z_plugin(self, archive_path: Path, special: str | None) -> bool:
        """Extract a 7z plugin to VS plugins directory."""
        try:
            # Try to find 7z executable
            seven_zip = self._find_7z()
            if not seven_zip:
                self.progress_signal.emit("  Error: 7-Zip not found. Please install 7-Zip.")
                return False

            with tempfile.TemporaryDirectory() as temp_dir:
                # Extract with 7z
                result = subprocess.run(
                    [seven_zip, "x", str(archive_path), f"-o{temp_dir}", "-y"],
                    capture_output=True,
                    text=True,
                    timeout=300,
                )
                if result.returncode != 0:
                    self.progress_signal.emit(f"  7z error: {result.stderr}")
                    return False

                temp_path = Path(temp_dir)

                # Debug: list all DLLs found
                all_dlls = list(temp_path.rglob("*.dll"))
                self.progress_signal.emit(f"  Found {len(all_dlls)} DLL(s) in archive")

                if special == "copy_fmtconv":
                    # Find and copy fmtconv.dll (usually in win64 subfolder)
                    for dll in temp_path.rglob("**/win64/fmtconv.dll"):
                        shutil.copy2(dll, VS_PLUGINS_DIR / "fmtconv.dll")
                        return True
                    # Fallback: search anywhere
                    for dll in temp_path.rglob("fmtconv.dll"):
                        shutil.copy2(dll, VS_PLUGINS_DIR / "fmtconv.dll")
                        return True
                    self.progress_signal.emit(f"  fmtconv.dll not found in archive")
                    return False
                elif special == "copy_imwri":
                    # Find and copy libimwri.dll from win64 folder
                    for dll in temp_path.rglob("**/win64/libimwri.dll"):
                        shutil.copy2(dll, VS_PLUGINS_DIR / "libimwri.dll")
                        self.progress_signal.emit(f"  Copied libimwri.dll from win64")
                        return True
                    # Fallback: search anywhere for libimwri.dll
                    for dll in temp_path.rglob("libimwri.dll"):
                        shutil.copy2(dll, VS_PLUGINS_DIR / "libimwri.dll")
                        self.progress_signal.emit(f"  Copied libimwri.dll")
                        return True
                    # Try imwri.dll as last resort
                    for dll in temp_path.rglob("imwri.dll"):
                        shutil.copy2(dll, VS_PLUGINS_DIR / "imwri.dll")
                        self.progress_signal.emit(f"  Copied imwri.dll")
                        return True
                    # Show what DLLs were found
                    self.progress_signal.emit(f"  DLLs in archive: {[d.name for d in all_dlls]}")
                    return False
                else:
                    # Copy all DLL files
                    for dll in temp_path.rglob("*.dll"):
                        shutil.copy2(dll, VS_PLUGINS_DIR / dll.name)
                    return True
        except Exception as e:
            self.progress_signal.emit(f"  Extract error: {e}")
            return False

    def _extract_7z_multi(self, parts: list[Path]) -> bool:
        """Extract multi-part 7z archive, preserving folder structure."""
        try:
            seven_zip = self._find_7z()
            if not seven_zip:
                self.progress_signal.emit("  Error: 7-Zip not found. Please install 7-Zip.")
                return False

            # Sort parts to ensure .001 comes first
            parts = sorted(parts, key=lambda p: p.suffix)
            first_part = parts[0]

            # Extract directly to VS_PLUGINS_DIR, preserving folder structure
            result = subprocess.run(
                [seven_zip, "x", str(first_part), f"-o{VS_PLUGINS_DIR}", "-y"],
                capture_output=True,
                text=True,
                timeout=600,
            )
            if result.returncode != 0:
                self.progress_signal.emit(f"  7z error: {result.stderr}")
                return False

            self.progress_signal.emit(f"  Extracted to: {VS_PLUGINS_DIR}")
            return True
        except Exception as e:
            self.progress_signal.emit(f"  Extract error: {e}")
            return False

    def _find_7z(self) -> str | None:
        """Find 7-Zip executable."""
        # Common locations
        locations = [
            r"C:\Program Files\7-Zip\7z.exe",
            r"C:\Program Files (x86)\7-Zip\7z.exe",
            shutil.which("7z"),
        ]
        for loc in locations:
            if loc and Path(loc).exists():
                return loc
        return None

    def _download_and_extract_tool(
        self, name: str, url: str, extract_type: str
    ) -> bool:
        """Download and extract an external tool to TOOLS_DIR."""
        temp_file = self._download_file(url, name)
        if not temp_file:
            return False

        try:
            with zipfile.ZipFile(temp_file, "r") as zf:
                # Extract directly to TOOLS_DIR ("extract here" style)
                zf.extractall(TOOLS_DIR)
            return True
        except Exception as e:
            self.progress_signal.emit(f"  Extract error: {e}")
            return False
        finally:
            try:
                temp_file.unlink()
            except Exception:
                pass

    def _add_to_path(self, directory: Path) -> bool:
        """Add a directory to the user's PATH environment variable."""
        try:
            import winreg

            # Open the Environment key
            with winreg.OpenKey(
                winreg.HKEY_CURRENT_USER,
                r"Environment",
                0,
                winreg.KEY_READ | winreg.KEY_WRITE,
            ) as key:
                try:
                    current_path, _ = winreg.QueryValueEx(key, "Path")
                except FileNotFoundError:
                    current_path = ""

                dir_str = str(directory)
                if dir_str.lower() not in current_path.lower():
                    if current_path:
                        new_path = f"{current_path};{dir_str}"
                    else:
                        new_path = dir_str
                    winreg.SetValueEx(key, "Path", 0, winreg.REG_EXPAND_SZ, new_path)

                    # Broadcast environment change
                    try:
                        import ctypes
                        HWND_BROADCAST = 0xFFFF
                        WM_SETTINGCHANGE = 0x001A
                        SMTO_ABORTIFHUNG = 0x0002
                        ctypes.windll.user32.SendMessageTimeoutW(
                            HWND_BROADCAST,
                            WM_SETTINGCHANGE,
                            0,
                            "Environment",
                            SMTO_ABORTIFHUNG,
                            5000,
                            None,
                        )
                    except Exception:
                        pass

                    return True
                else:
                    self.progress_signal.emit(f"  {dir_str} already in PATH")
                    return True
        except Exception as e:
            self.progress_signal.emit(f"  PATH error: {e}")
            return False


class DependenciesWindow(QDialog):
    """
    Dependencies installation window.

    Provides UI for installing all required dependencies:
    - Python packages via pip
    - VapourSynth plugins
    - External tools (ffmpeg, gifski, avifenc, pngquant, pingo)
    """

    def __init__(self, parent: QWidgetType | None = None):
        super().__init__(parent)
        self.setWindowTitle("Dependencies")
        self.setMinimumSize(600, 500)
        self.resize(700, 600)

        self._worker: InstallWorker | None = None
        self._animation_timer: QTimer | None = None
        self._animation_dots = 0

        self._build_ui()
        self._connect_signals()

    def _build_ui(self) -> None:
        """Build the dialog UI."""
        layout = QVBoxLayout()
        self.setLayout(layout)

        # Info section
        info_group = QGroupBox("Required Dependencies")
        info_layout = QVBoxLayout()
        info_group.setLayout(info_layout)

        info_text = QLabel(
            "This will install all dependencies required for VapourSynth Image Upscaler:\n\n"
            "<b>Python packages:</b> PySide6, vsjetpack, Pillow, numpy\n\n"
            "<b>VapourSynth plugins:</b> akarin, fmtconv, resize2, vapoursynth-zip, zsmooth, "
            "bestsource, imwri, deblock, dctfilter, vsmlrt\n\n"
            "<b>External tools:</b> ffmpeg, gifski, avifenc (animated output), "
            "pngquant, pingo (PNG optimization)\n\n"
            "<b>Note:</b> 7-Zip is required for extracting .7z files.\n"
            "VapourSynth must be installed manually."
        )
        info_text.setTextFormat(Qt.RichText)
        info_text.setWordWrap(True)
        info_layout.addWidget(info_text)

        layout.addWidget(info_group)

        # Progress section
        progress_group = QGroupBox("Installation Progress")
        progress_layout = QVBoxLayout()
        progress_group.setLayout(progress_layout)

        self._status_label = QLabel("Ready to install")
        progress_layout.addWidget(self._status_label)

        self._progress_bar = QProgressBar()
        self._progress_bar.setRange(0, 100)
        self._progress_bar.setValue(0)
        progress_layout.addWidget(self._progress_bar)

        # Log output
        self._log_output = QTextEdit()
        self._log_output.setReadOnly(True)
        self._log_output.setMinimumHeight(200)
        self._log_output.setStyleSheet("font-family: Consolas, monospace; font-size: 10pt;")
        progress_layout.addWidget(self._log_output)

        layout.addWidget(progress_group)

        # Buttons
        btn_layout = QHBoxLayout()

        self._install_button = QPushButton("Install All Dependencies")
        self._install_button.setMinimumHeight(40)
        self._install_button.setStyleSheet("font-weight: bold;")
        btn_layout.addWidget(self._install_button, 2)

        self._cancel_button = QPushButton("Cancel")
        self._cancel_button.setEnabled(False)
        btn_layout.addWidget(self._cancel_button, 1)

        self._close_button = QPushButton("Close")
        btn_layout.addWidget(self._close_button, 1)

        layout.addLayout(btn_layout)

    def _connect_signals(self) -> None:
        """Connect widget signals."""
        self._install_button.clicked.connect(self._on_install_clicked)
        self._cancel_button.clicked.connect(self._on_cancel_clicked)
        self._close_button.clicked.connect(self.close)

    def _on_install_clicked(self) -> None:
        """Start the installation process."""
        self._install_button.setEnabled(False)
        self._cancel_button.setEnabled(True)
        self._close_button.setEnabled(False)
        self._log_output.clear()
        self._progress_bar.setValue(0)

        # Start animation
        self._animation_dots = 0
        self._animation_timer = QTimer(self)
        self._animation_timer.timeout.connect(self._update_animation)
        self._animation_timer.start(500)

        # Start worker
        self._worker = InstallWorker(
            install_pip=True,
            install_plugins=True,
            install_tools=True,
        )
        self._worker.progress_signal.connect(self._on_progress)
        self._worker.status_signal.connect(self._on_status)
        self._worker.finished_signal.connect(self._on_finished)
        self._worker.start()

    def _on_cancel_clicked(self) -> None:
        """Cancel the installation."""
        if self._worker:
            self._worker.cancel()
            self._status_label.setText("Cancelling...")

    def _update_animation(self) -> None:
        """Update the loading animation."""
        self._animation_dots = (self._animation_dots + 1) % 4
        dots = "." * self._animation_dots
        current = self._status_label.text().rstrip(".")
        if current:
            base = current.split("...")[0].split("..")[0].split(".")[0]
            self._status_label.setText(f"{base}{dots}")

    def _on_progress(self, message: str) -> None:
        """Handle progress log message."""
        self._log_output.append(message)
        # Auto-scroll to bottom
        scrollbar = self._log_output.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def _on_status(self, status: str, current: int, total: int) -> None:
        """Handle status update."""
        self._status_label.setText(status)
        if total > 0:
            percent = int(100 * current / total)
            self._progress_bar.setValue(percent)

    def _on_finished(self, success: bool, message: str) -> None:
        """Handle installation completion."""
        if self._animation_timer:
            self._animation_timer.stop()
            self._animation_timer = None

        self._install_button.setEnabled(True)
        self._cancel_button.setEnabled(False)
        self._close_button.setEnabled(True)
        self._progress_bar.setValue(100 if success else 0)
        self._status_label.setText(message)

        if success:
            QMessageBox.information(
                self,
                "Installation Complete",
                "All dependencies have been installed successfully!\n\n"
                "Note: You may need to restart the application for changes to take effect.",
            )
        else:
            QMessageBox.warning(
                self,
                "Installation Issue",
                f"{message}\n\nCheck the log output for details.",
            )

        self._worker = None

    def closeEvent(self, event) -> None:
        """Handle window close."""
        if self._worker and self._worker.isRunning():
            reply = QMessageBox.question(
                self,
                "Installation in Progress",
                "Installation is still in progress. Are you sure you want to close?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if reply == QMessageBox.No:
                event.ignore()
                return
            self._worker.cancel()
            self._worker.wait(5000)

        if self._animation_timer:
            self._animation_timer.stop()

        super().closeEvent(event)
