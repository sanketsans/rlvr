import logging
import sys

def setup_logger(
    name: str = "qwen3_rlvr",
    level: int = logging.INFO,
    fmt: str = "[%(asctime)s] %(levelname)s %(name)s: %(message)s",
    datefmt: str = "%Y-%m-%d %H:%M:%S",
    stream = sys.stdout,
) -> logging.Logger:
    """
    Set up a consistent logger to be used in place of print statements.

    Usage:
        from qwen3_rlvr.logging.logger import setup_logger
        logger = setup_logger(__name__)
        logger.info("This is an info message")
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    # Prevent adding multiple handlers in notebook or multi-import envs
    if not logger.handlers:
        handler = logging.StreamHandler(stream)
        formatter = logging.Formatter(fmt, datefmt=datefmt)
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.propagate = False
    return logger

# Prefer this pattern everywhere:
# logger = setup_logger(__name__)
# logger.info("Message")
# logger.warning("Warning")
# logger.error("Error")
# logger.debug("Debug message")