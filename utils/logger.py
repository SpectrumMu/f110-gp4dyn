import logging
import os
from datetime import datetime

logging.basicConfig(level=logging.INFO)

LOSS_LEVEL = 25  # Between INFO (20) and WARNING (30)
logging.addLevelName(LOSS_LEVEL, "LOSS")

def loss(self, message, *args, **kwargs):
    if self.isEnabledFor(LOSS_LEVEL):
        self._log(LOSS_LEVEL, message, args, **kwargs)

logging.Logger.loss = loss

class NotLossFilter(logging.Filter):
    def filter(self, record):
        return record.levelno != LOSS_LEVEL


def setup_logger(name="train_logger", log_dir="logs"):
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)  # Capture all messages
    logger.propagate = False        # Prevent duplication

    if not logger.handlers:
        formatter = logging.Formatter('[%(asctime)s] %(levelname)s - %(message)s')

        # File handler: write all logs including DEBUG (loss)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        file_handler.setLevel(logging.DEBUG)
        logger.addHandler(file_handler)

        # Console handler: write only INFO and above (no DEBUG/loss)
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        console_handler.setLevel(logging.INFO)
        logger.addHandler(console_handler)

    return logger