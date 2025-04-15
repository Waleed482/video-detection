

import sqlite3
import csv
import numpy as np
import cv2
from threading import Thread, Lock
from queue import Queue
import time
import logging
from datetime import datetime

class FrameStorage:
    def __init__(self, db_path, csv_file):
        self.db_path = db_path
        self.csv_file = csv_file
        self.lock = Lock()
        self.running = True
        self.frame_queue = Queue(maxsize=10)
        self.store_thread = Thread(target=self._store_worker)
        self.store_thread.daemon = True
        self.store_thread.start()

    def _store_worker(self):
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        while self.running:
            if not self.frame_queue.empty():
                frame, timestamp = self.frame_queue.get()
                try:
                    _, buffer = cv2.imencode('.jpg', frame)
                    frame_binary = buffer.tobytes()
                    c = conn.cursor()
                    c.execute("INSERT INTO frames (timestamp, frame) VALUES (?, ?)",
                              (timestamp, frame_binary))
                    frame_id = c.lastrowid
                    conn.commit()
                    with self.lock:
                        with open(self.csv_file, 'a', newline='') as f:
                            writer = csv.writer(f)
                            writer.writerow([frame_id, timestamp])
                    logging.info(f"Stored frame {frame_id} at {timestamp}")
                except Exception as e:
                    logging.error(f"Error storing frame: {e}")
            time.sleep(0.05)
        conn.close()

    def store_frame(self, frame, timestamp):
        if self.frame_queue.full():
            self.frame_queue.get()
        self.frame_queue.put((frame, timestamp))

    def get_latest_frame(self):
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        try:
            c = conn.cursor()
            c.execute("SELECT frame FROM frames ORDER BY id DESC LIMIT 1")
            result = c.fetchone()
            if result:
                frame_binary = result[0]
                nparr = np.frombuffer(frame_binary, np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                return frame
            return None
        except Exception as e:
            logging.error(f"Error retrieving frame: {e}")
            return None
        finally:
            conn.close()

    def stop(self):
        self.running = False
        self.store_thread.join()

class DescriptionStorage:
    def __init__(self, db_path, csv_file):
        self.db_path = db_path
        self.csv_file = csv_file
        self.lock = Lock()
        self.running = True
        self.description_queue = Queue(maxsize=10)
        self.store_thread = Thread(target=self._store_worker)
        self.store_thread.daemon = True
        self.store_thread.start()
        
        # Create timestamp index
        conn = sqlite3.connect(self.db_path)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON descriptions(timestamp)")
        conn.commit()
        conn.close()

    def _store_worker(self):
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        while self.running:
            if not self.description_queue.empty():
                description, timestamp = self.description_queue.get()
                try:
                    c = conn.cursor()
                    c.execute("INSERT INTO descriptions (timestamp, description) VALUES (?, ?)",
                              (timestamp, description))
                    description_id = c.lastrowid
                    conn.commit()
                    with self.lock:
                        with open(self.csv_file, 'a', newline='') as f:
                            writer = csv.writer(f)
                            writer.writerow([description_id, timestamp, description])
                    logging.info(f"Stored description {description_id} at {timestamp}: {description}")
                except Exception as e:
                    logging.error(f"Error storing description: {e}")
            time.sleep(0.05)
        conn.close()

    def store_description(self, description, timestamp):
        if self.description_queue.full():
            self.description_queue.get()
        self.description_queue.put((description, timestamp))

    def get_latest_description(self):
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        try:
            c = conn.cursor()
            c.execute("SELECT description, timestamp FROM descriptions ORDER BY id DESC LIMIT 1")
            result = c.fetchone()
            if result:
                description, timestamp = result
                return description, timestamp
            return None, None
        except Exception as e:
            logging.error(f"Error retrieving description: {e}")
            return None, None
        finally:
            conn.close()
        
    def stop(self):
        self.running = False
        self.store_thread.join()