"""
PrognosX — Logging Utility
============================
Provides a single `get_logger()` factory so every module logs
consistently to both console and a rotating file under logs/.
"""

import logging
import sys
from datetime import datetime
from logging.handlers import RotatingFileHandler

from src.config.config import LOG_DIR, LOG_LEVEL

_LOG_FORMAT = "[%(asctime)s] %(levelname)-8s | %(name)s | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

_LOG_FILE = LOG_DIR / f"prognosx_{datetime.now().strftime('%Y%m%d')}.log"


def get_logger(name: str = "prognosx") -> logging.Logger:
    """Return a configured logger.

    Parameters
    ----------
    name : str
        Usually `__name__` of the calling module.
    """
    logger = logging.getLogger(name)

    if logger.handlers:
        # Already configured (avoids duplicate handlers on re-import)
        return logger

    logger.setLevel(LOG_LEVEL)

    formatter = logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT)

    console_handler = logging.StreamHandler(sys.stdout) # handler sends logs to the terminal
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # It prevents the log file from growing indefinitely
    file_handler = RotatingFileHandler(
        _LOG_FILE, maxBytes=5 * 1024 * 1024, backupCount=5
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    logger.propagate = False
    return logger
