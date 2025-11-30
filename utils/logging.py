import logging
import logging.handlers
import os
import gzip # Added import
from typing import Optional


_LOGGER_CONFIGURED = False
_LOG_FILE_PATH: Optional[str] = None


def get_default_log_path() -> str:
    # Repository root is parent of this utils package
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    log_dir = os.path.join(base_dir, 'logs')
    os.makedirs(log_dir, exist_ok=True)
    return os.path.join(log_dir, 'app.log')


def setup_logging(log_file: Optional[str] = None, level: int = logging.INFO) -> None:
    """Configure root logger with a rotating file handler and console handler.

    Call this once at application start. Subsequent calls are no-ops.
    """
    global _LOGGER_CONFIGURED, _LOG_FILE_PATH
    if _LOGGER_CONFIGURED:
        return

    if log_file is None:
        log_file = get_default_log_path()

    _LOG_FILE_PATH = os.path.abspath(log_file)

    root = logging.getLogger()
    root.setLevel(level)

    # Use a concise, human-friendly format with millisecond precision and
    # short logger context. File and console use the same timestamp formatting
    # for consistency.
    datefmt = '%Y-%m-%d %H:%M:%S'
    # Keep file logs concise: timestamp (ms), level, message. Remove module:lineno.
    file_fmt = logging.Formatter('%(asctime)s.%(msecs)03d | %(levelname)-5s | %(message)s', datefmt=datefmt)

    # Timed rotating file handler (hourly) keeps recent history in logs/app.log
    fh = logging.handlers.TimedRotatingFileHandler(_LOG_FILE_PATH, when='H', interval=1, backupCount=24, encoding='utf-8')
    fh.setLevel(level)
    fh.setFormatter(file_fmt)
    root.addHandler(fh)

    # Console handler uses a slightly more compact formatter (no lineno)
    # Console also omits logger name/lineno for a cleaner runtime view.
    console_fmt = logging.Formatter('%(asctime)s | %(levelname)-5s | %(message)s', datefmt=datefmt)
    ch = logging.StreamHandler()
    ch.setLevel(level)
    ch.setFormatter(console_fmt)
    root.addHandler(ch)

    _LOGGER_CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Return a logger with the given name. Call `setup_logging()` first in app entrypoints."""
    return logging.getLogger(name)


def get_log_file_path() -> Optional[str]:
    return _LOG_FILE_PATH or get_default_log_path()


def tail_log(log_file_path: Optional[str] = None, lines: int = 200) -> str:
    """Return last `lines` lines of the specified log file as a single string.
    If `log_file_path` is None, it defaults to the main log file.
    Handles gzipped log files automatically.
    """
    path = log_file_path if log_file_path is not None else get_log_file_path()
    if not path:
        return "No log file path specified or default not found."

    try:
        if path.endswith('.gz'):
            with gzip.open(path, 'rt', encoding='utf-8', errors='replace') as f:
                content = f.read().splitlines()
        else:
            with open(path, 'r', encoding='utf-8', errors='replace') as f:
                content = f.read().splitlines()
        
        if lines <= 0:
            return '\n'.join(content)
        return '\n'.join(content[-lines:])
    except FileNotFoundError:
        return f"Log file not found: {path}"
    except Exception as e:
        return f"Could not read log file {path}: {e}"
