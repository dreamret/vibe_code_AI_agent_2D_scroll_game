"""Local AI Player-Action Simulator - entry point."""
import sys


def check_dependencies() -> list[str]:
    missing = []
    for mod, pip in (("PySide6", "PySide6"), ("cv2", "opencv-python"), ("numpy", "numpy"),
                     ("mss", "mss"), ("pynput", "pynput"), ("tinydb", "tinydb")):
        try:
            __import__(mod)
        except ImportError:
            missing.append(pip)
    return missing


def main() -> int:
    missing = check_dependencies()
    if missing:
        print("Missing packages:", ", ".join(missing))
        print("Run start.bat (or: pip install -r requirements.txt)")
        return 2

    from PySide6.QtWidgets import QApplication, QMessageBox
    from app import APP_NAME
    from app.controller import AppController
    from app.gui.main_window import MainWindow

    qt = QApplication(sys.argv)
    qt.setApplicationName(APP_NAME)
    try:
        ctrl = AppController()
    except Exception as exc:                       # e.g. database cannot be opened
        QMessageBox.critical(None, "Startup failed", str(exc))
        return 1
    win = MainWindow(ctrl)
    if not ctrl.settings["notice_accepted"]:
        if not win.show_notice(first_run=True):
            ctrl.shutdown()
            return 0
        ctrl.settings["notice_accepted"] = True
        ctrl.settings.save()
    win.show()
    return qt.exec()


if __name__ == "__main__":
    sys.exit(main())
