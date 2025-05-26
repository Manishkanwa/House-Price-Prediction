# logger.py
from loguru import logger
import os

LOG_FILE = os.path.join("logs", "pipeline.log")
logger.add(LOG_FILE, rotation="500 KB", level="INFO")

def log_prediction(house_id, prediction):
    logger.info(f"Prediction made for House ID {house_id}: ₹{prediction:,.2f}")
