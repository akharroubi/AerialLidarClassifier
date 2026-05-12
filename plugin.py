"""Main plugin class for Aerial LiDAR Classifier.

The plugin contributes:
  - a checkable toolbar button + menu entry that toggles the main dock
    panel (right-side, QGIS-native style),
  - a Processing provider registered with the QGIS Processing registry,
  - menu actions for opening the documentation, the About dialog and
    for clearing GPU memory.

All heavy imports (torch, laspy...) are deferred so loading the plugin
is cheap and does not depend on the AI stack being installed.
"""

import os
import webbrowser

from qgis.PyQt.QtCore import Qt
from qgis.PyQt.QtGui import QIcon
from qgis.PyQt.QtWidgets import QAction
from qgis.core import QgsApplication, Qgis

from . import tr
from .config import PLUGIN_NAME
from .utils.logger import log_info, log_warning


MENU_LABEL = "&Aerial LiDAR Classifier"
DOCS_URL = "https://github.com/akharroubi/AerialLidarClassifier"


class AerialLidarClassifierPlugin:
    """QGIS plugin entry point."""

    def __init__(self, iface):
        self.iface = iface
        self.plugin_dir = os.path.dirname(__file__)
        self.icon_path = os.path.join(self.plugin_dir, "icon.png")

        self.actions = []
        self.toolbar = None

        # Dock + provider are created lazily so the AI stack is only
        # touched when the user actually opens the plugin.
        self.dock = None
        self.toggle_action = None
        self.provider = None

        # Dependency-install dock + worker (lazy).
        self._deps_dock = None
        self._deps_worker = None
        self._deps_available = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def initGui(self):  # noqa: N802 - QGIS API
        self.toolbar = self.iface.addToolBar(PLUGIN_NAME)
        self.toolbar.setObjectName("AerialLidarClassifierToolbar")

        # Main toggle action (checkable - shows/hides the dock panel)
        self.toggle_action = self._add_action(
            QIcon(self.icon_path),
            tr("Aerial LiDAR Classifier"),
            self.toggle_dock,
            checkable=True,
            add_to_toolbar=True,
            tooltip=tr("Show / hide the Aerial LiDAR Classifier panel"),
        )
        self.iface.addPluginToMenu(MENU_LABEL, self.toggle_action)

        # Secondary actions (menu only)
        clear_gpu = self._add_action(
            QIcon(":/images/themes/default/mActionRefresh.svg"),
            tr("Clear GPU memory"),
            self.clear_gpu_memory,
            tooltip=tr("Release cached GPU memory (CUDA only)"),
        )
        self.iface.addPluginToMenu(MENU_LABEL, clear_gpu)

        help_action = self._add_action(
            QIcon(":/images/themes/default/mActionHelpContents.svg"),
            tr("Help / Documentation"),
            self._open_docs,
        )
        self.iface.addPluginToMenu(MENU_LABEL, help_action)

        about_action = self._add_action(
            QIcon(":/images/themes/default/mActionPropertiesWidget.svg"),
            tr("About..."),
            self._show_about,
        )
        self.iface.addPluginToMenu(MENU_LABEL, about_action)

        # Processing provider
        try:
            from .processing.provider import AerialLidarProvider
            self.provider = AerialLidarProvider()
            QgsApplication.processingRegistry().addProvider(self.provider)
        except Exception as exc:
            log_warning(f"Could not register Processing provider: {exc}")
            self.provider = None

    def unload(self):
        # Stop any in-flight install worker
        if self._deps_worker and self._deps_worker.isRunning():
            try:
                self._deps_worker.cancel()
                self._deps_worker.terminate()
                self._deps_worker.wait(5000)
            except Exception:
                pass
        self._deps_worker = None

        # Tear down the deps dock if it's still open
        if self._deps_dock is not None:
            try:
                self.iface.removeDockWidget(self._deps_dock)
                self._deps_dock.deleteLater()
            except Exception:
                pass
            self._deps_dock = None

        # Tear down the main dock (it owns running tasks)
        if self.dock is not None:
            try:
                self.iface.removeDockWidget(self.dock)
                self.dock.deleteLater()
            except Exception:
                pass
            self.dock = None

        # Remove menu / toolbar items
        for action in self.actions:
            try:
                self.iface.removePluginMenu(MENU_LABEL, action)
            except Exception:
                pass
            if self.toolbar:
                self.toolbar.removeAction(action)

        if self.toolbar:
            del self.toolbar
            self.toolbar = None

        # Unregister Processing provider
        if self.provider:
            try:
                QgsApplication.processingRegistry().removeProvider(self.provider)
            except Exception:
                pass
            self.provider = None

        self.actions = []
        self.toggle_action = None

    # ------------------------------------------------------------------
    # Dock toggle (with dependency gate)
    # ------------------------------------------------------------------

    def toggle_dock(self):
        """Show the dock (after dependency check) or hide it if visible."""
        if self.dock is not None and self.dock.isVisible():
            self.dock.hide()
            return

        if not self._ensure_dependencies():
            # Setup dock has been shown; the toggle action's checked
            # state was reset there.
            return

        self._create_dock_if_needed()
        self.dock.show()
        self.dock.raise_()

    # ------------------------------------------------------------------
    # Dependency gating (isolated per-user virtual environment)
    # ------------------------------------------------------------------

    def _ensure_dependencies(self) -> bool:
        """Return True iff the isolated venv has all packages ready.

        Logs every decision path so problems are diagnosable from the
        QGIS Log panel (filter on the "Aerial LiDAR Classifier" tag).
        """
        if self._deps_available:
            log_info("Dependency check: already marked ready this session.")
            return True

        try:
            from .utils.venv_manager import (
                ensure_venv_packages_available,
                get_venv_status,
            )
            is_ready, status_msg = get_venv_status()
            log_info(
                f"Dependency check: get_venv_status -> "
                f"ready={is_ready} msg={status_msg}"
            )
            if is_ready:
                ok = ensure_venv_packages_available()
                log_info(
                    f"Dependency check: ensure_venv_packages_available "
                    f"returned {ok}"
                )
                if ok:
                    self._deps_available = True
                    # If a stale setup dock from a previous session is
                    # still hanging around, dispose of it now.
                    self._close_deps_dock()
                    return True
        except Exception as exc:
            log_warning(
                f"Dependency check raised an exception: {exc}. "
                "Falling back to the Setup dock."
            )

        log_info("Dependency check: opening the Setup dock.")
        self._show_deps_dock()
        if self.toggle_action:
            self.toggle_action.setChecked(False)
        return False

    def _close_deps_dock(self) -> None:
        """Tear down the setup dock (no-op if not open)."""
        if self._deps_dock is None:
            return
        try:
            self.iface.removeDockWidget(self._deps_dock)
            self._deps_dock.deleteLater()
        except Exception:
            pass
        self._deps_dock = None

    def _show_deps_dock(self) -> None:
        """Create and show the setup dock (idempotent)."""
        if self._deps_dock is not None:
            self._deps_dock.show()
            self._deps_dock.raise_()
            return

        from .dialogs.deps_install_dialog import DepsInstallDockWidget

        self._deps_dock = DepsInstallDockWidget(self.iface.mainWindow())
        self._deps_dock.install_requested.connect(self._on_install_requested)
        self._deps_dock.cancel_requested.connect(self._on_cancel_install)
        self.iface.addDockWidget(
            Qt.DockWidgetArea.RightDockWidgetArea, self._deps_dock
        )
        self._deps_dock.show()
        self._deps_dock.raise_()

    def _on_install_requested(self) -> None:
        """Kick off the background install worker."""
        if self._deps_worker and self._deps_worker.isRunning():
            return

        from .utils.venv_manager import detect_nvidia_gpu
        from .workers.deps_install_worker import DepsInstallWorker

        has_gpu, _ = detect_nvidia_gpu()
        self._deps_worker = DepsInstallWorker(cuda_enabled=has_gpu)
        self._deps_worker.progress.connect(self._on_install_progress)
        self._deps_worker.finished.connect(self._on_install_finished)
        if self._deps_dock:
            self._deps_dock.show_progress_ui()
        self._deps_worker.start()

    def _on_install_progress(self, percent: int, message: str) -> None:
        if self._deps_dock:
            self._deps_dock.set_progress(percent, message)

    def _on_install_finished(self, success: bool, message: str) -> None:
        log_info(f"Install finished: success={success} - {message}")

        if self._deps_dock:
            self._deps_dock.show_complete_ui(success, message)

        if not success:
            # Leave the setup dock visible with the failure message so
            # the user can read it and click 'Reinstall' if they want.
            return

        # Make the venv site-packages importable, then open the main dock.
        # If anything goes wrong here we still close the setup dock and
        # let the user re-toggle; the next click will re-evaluate the
        # state and either open the main dock or re-open setup.
        try:
            from .utils.venv_manager import ensure_venv_packages_available
            ok = ensure_venv_packages_available()
            if not ok:
                log_warning(
                    "ensure_venv_packages_available returned False - "
                    "venv site-packages could not be located"
                )
                return
            self._deps_available = True
        except Exception as exc:
            log_warning(f"Could not load venv packages: {exc}")
            return

        # Success path: dispose of the setup dock and open the main dock.
        self._close_deps_dock()
        self._create_dock_if_needed()
        self.dock.show()
        self.dock.raise_()
        if self.toggle_action:
            self.toggle_action.setChecked(True)

    def _on_cancel_install(self) -> None:
        if self._deps_worker and self._deps_worker.isRunning():
            self._deps_worker.cancel()
        if self._deps_dock:
            self._deps_dock.show_install_ui()

    # ------------------------------------------------------------------

    def _create_dock_if_needed(self):
        if self.dock is not None:
            return
        from .gui.main_panel import ClassifierDockWidget
        self.dock = ClassifierDockWidget(self.iface, self.iface.mainWindow())
        self.dock.visibilityChanged.connect(self._on_dock_visibility_changed)
        self.iface.addDockWidget(
            Qt.DockWidgetArea.RightDockWidgetArea, self.dock)

    def _on_dock_visibility_changed(self, visible: bool):
        if self.toggle_action:
            self.toggle_action.setChecked(visible)

    # ------------------------------------------------------------------
    # Secondary actions
    # ------------------------------------------------------------------

    def clear_gpu_memory(self):
        """Release cached GPU memory (no-op on CPU-only installs)."""
        try:
            # Make the venv packages importable into QGIS Python first.
            from .utils.venv_manager import ensure_venv_packages_available
            ensure_venv_packages_available()
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                self.iface.messageBar().pushMessage(
                    PLUGIN_NAME, tr("GPU memory cache cleared."),
                    level=Qgis.Success, duration=4,
                )
                log_info("GPU memory cache cleared via menu action.")
            else:
                self.iface.messageBar().pushMessage(
                    PLUGIN_NAME, tr("No CUDA GPU detected."),
                    level=Qgis.Info, duration=4,
                )
        except Exception as exc:
            self.iface.messageBar().pushMessage(
                PLUGIN_NAME,
                tr("Could not clear GPU memory: {err}").format(err=exc),
                level=Qgis.Warning, duration=6,
            )

    def _open_docs(self):
        webbrowser.open(DOCS_URL)

    def _show_about(self):
        from .dialogs.about_dialog import AboutDialog
        AboutDialog(self.iface.mainWindow()).exec()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _add_action(
        self,
        icon: QIcon,
        text: str,
        callback,
        *,
        checkable: bool = False,
        add_to_toolbar: bool = False,
        tooltip: str = None,
    ) -> QAction:
        action = QAction(icon, text, self.iface.mainWindow())
        action.setCheckable(checkable)
        action.triggered.connect(callback)
        if tooltip:
            action.setToolTip(tooltip)
            action.setStatusTip(tooltip)
        # macOS: keep the action under the plugin menu (do not auto-route
        # to the application menu).
        try:
            action.setMenuRole(QAction.MenuRole.NoRole)
        except AttributeError:
            try:
                action.setMenuRole(QAction.NoRole)
            except Exception:
                pass
        if add_to_toolbar and self.toolbar:
            self.toolbar.addAction(action)
        self.actions.append(action)
        return action
