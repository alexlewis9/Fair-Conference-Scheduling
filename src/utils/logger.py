# logger_config.py
import logging
import os
from datetime import datetime

def setup_session_logger(log_path: str) -> logging.Logger:
    logger = logging.getLogger()  # root logger
    logger.setLevel(logging.INFO)

    # Clear existing file handlers
    logger.handlers = [h for h in logger.handlers if not isinstance(h, logging.FileHandler)]

    # Add new file handler
    file_handler = logging.FileHandler(log_path)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger, file_handler  # Return handler for later cleanup
