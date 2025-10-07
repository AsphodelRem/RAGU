import logging
import sys

import openai

openai._utils._logs.logger.setLevel(logging.WARNING)
openai._utils._logs.httpx_logger.setLevel(logging.WARNING)

from loguru import logger

logger.remove()
logger.add(
    sys.stdout,
    colorize=True,
    enqueue=True,
    level="INFO",
    format="<cyan>{time:HH:mm:ss}</cyan> | <level>{level: <8}</level> | <level>{message}</level>"
)

__all__ = ["logger"]
