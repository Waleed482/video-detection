import sqlite3
import csv
import torch
import logging
from transformers import AutoProcessor, BlipForConditionalGeneration

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# SQLite Database Setup for Frames and Descriptions
def init_database():
    conn = sqlite3.connect('frames.db', check_same_thread=False)
    c = conn.cursor()
    # Table for frames
    c.execute('''CREATE TABLE IF NOT EXISTS frames
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  timestamp TEXT,
                  frame BLOB)''')
    # Table for descriptions
    c.execute('''CREATE TABLE IF NOT EXISTS descriptions
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  timestamp TEXT,
                  description TEXT)''')
    conn.commit()
    return conn

# CSV Setup for Frames
def init_csv():
    csv_file = 'frames_log.csv'
    with open(csv_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['frame_id', 'timestamp'])
    return csv_file

# CSV Setup for Descriptions
def init_description_csv():
    csv_file = 'descriptions_log.csv'
    with open(csv_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['description_id', 'timestamp', 'description'])
    return csv_file

def get_gpu_usage():
    if torch.cuda.is_available():
        mem_alloc = torch.cuda.memory_allocated() / (1024 ** 2)
        mem_total = torch.cuda.get_device_properties(0).total_memory / (1024 ** 2)
        return f"GPU: {mem_alloc:.1f}/{mem_total:.1f} MB"
    return "Using CPU"

def load_models():
    try:
        processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if device == 'cuda':
            model = model.half().to(device)
        return processor, model, device
    except Exception as e:
        logging.error(f"Model load failed: {e}")
        return None, None, None