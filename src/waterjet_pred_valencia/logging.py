"""Configures loggers with various levels and coloring for this repo's modules."""

import logging
import sys

# List of modules which should use DEBUG instead of the default INFO log level.
DEBUG_MODULES = [
    "waterjet_pred_valencia.cli",
    "waterjet_pred_valencia.jet_state",
    "waterjet_pred_valencia.logging",
    "waterjet_pred_valencia.plotting",
    "waterjet_pred_valencia.tracer",
]

# To avoid ambiguity, use 'name' in lieu of 'module'
LOG_FMT = "[{asctime}] [{levelname}] [{module}] {message}"


class ColorFormatter(logging.Formatter):
    """Assigns colors to log levels."""

    # ANSI escape codes
    RESET = "\x1b[0m"
    COLORS = {
        logging.DEBUG: "\x1b[32m",  # green
        logging.INFO: "\x1b[37m",  # white/gray
        logging.WARNING: "\x1b[33m",  # yellow
        logging.ERROR: "\x1b[31m",  # red
        logging.CRITICAL: "\x1b[1;31m",  # bright/bold red
    }

    def format(self, record: logging.LogRecord) -> str:
        """Format one log record and colorize it based on its level."""
        message = super().format(record)
        color = self.COLORS.get(record.levelno, self.RESET)
        return f"{color}{message}{self.RESET}"


def configure_logging() -> None:
    """Configure logging.

    Call this function from the application's entrypoint and include the following line
    at the top of each module:

        logger = logging.getLogger(__name__)
    """

    # Undo global disable() if some library called it
    logging.disable(logging.NOTSET)

    # Make root logger noisy only at WARNING+ to reduce third-party spam
    root = logging.getLogger()
    root.setLevel(logging.WARNING)

    # Define logging handlers and formatting
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(ColorFormatter(LOG_FMT, style="{"))

    _setup_namespace_logger("waterjet_pred_valencia", logging.INFO, handler)

    for name in DEBUG_MODULES:
        logging.getLogger(name).setLevel(logging.DEBUG)

    return


def _setup_namespace_logger(
    name: str, level: int, handler: logging.StreamHandler
) -> None:
    """Attach the shared handler to a logger namespace.

    Args:
        name: Name of the logger.
        level: Log level (DEBUG/INFO/WARNING/ERROR/CRITICAL)
        handler: StreamHandler to attach
    """
    ns_logger = logging.getLogger(name)
    ns_logger.disabled = False
    ns_logger.setLevel(level)
    ns_logger.propagate = False
    ns_logger.handlers.clear()
    ns_logger.addHandler(handler)
    return
