"""
Utility function that binds a logger with project-specific settings.
"""

import sys

from typing import Optional

from loguru import logger
from loguru._logger import Logger

def bind_logger(_logger: Logger, log_path: Optional[str] = None) -> Logger:
    """
    Bind a logger with specific format and optional file logging.
    
    Parameters:
        _logger (Logger):
            The logger object to bind.
        log_path (Optional[str]):
            The path to log file.
    
    Returns:
        Logger:
            The configured logger object.
    """
    
    fmt = (
    "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
    "<level>{level: <8}</level> | "
    "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
    "[CSIT5210 {level:<8}]:\n\n<level>{message}</level>\n"
)
    _logger.remove()
    _logger.add(sys.stdout, format=fmt)

    if not log_path:
        return _logger
    
    _logger.add(log_path, rotation="10 MB", format=fmt)

    return _logger


if __name__ == "__main__":
    _logger = bind_logger(logger)

    _logger.info("test")
    _logger.warning("test")
    _logger.error("test")