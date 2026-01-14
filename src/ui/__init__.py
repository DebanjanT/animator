"""UI module - GTK and OpenCV backends"""

__all__ = ["MainWindow", "CVWindow", "run_cv_app"]

def __getattr__(name):
    """Lazy import to avoid GTK import errors when using OpenCV backend."""
    if name == "MainWindow":
        from .main_window import MainWindow
        return MainWindow
    elif name == "CVWindow":
        from .cv_window import CVWindow
        return CVWindow
    elif name == "run_cv_app":
        from .cv_window import run_cv_app
        return run_cv_app
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
