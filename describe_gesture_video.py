import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from tensorflow.keras.layers import TFSMLayer
import torch
import logging
import time
import sys
import speech_recognition as sr
import pyttsx3
from threading import Thread, Lock
from queue import Queue
from transformers import AutoProcessor, BlipForConditionalGeneration
import sqlite3
import csv
from datetime import datetime
import os

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize MediaPipe for gesture recognition
mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mpDraw = mp.solutions.drawing_utils

# Load gesture recognition model
gesture_model = TFSMLayer('mp_hand_gesture', call_endpoint='serving_default')
with open('gestures.names', 'r') as f:
    gesture_classNames = f.read().split('\n')
logger.info(f"Gesture class names: {gesture_classNames}")

# SQLite Database Setup
def init_database():
    conn = sqlite3.connect('frames.db', check_same_thread=False)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS frames
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  timestamp TEXT,
                  frame BLOB)''')
    conn.commit()
    return conn

# CSV Setup
def init_csv():
    csv_file = 'frames_log.csv'
    with open(csv_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['frame_id', 'timestamp'])
    return csv_file

# Speech Handler for listening to user queries
class SpeechHandler:
    def __init__(self):
        self.recognizer = sr.Recognizer()
        self.mic = sr.Microphone()
        self.current_speech_text = "System ready. Waiting for your command..."
        self.speech_queue = Queue(maxsize=1)
        self.lock = Lock()
        self.running = True
        self.is_listening = True
        self.thread = Thread(target=self._speech_worker)
        self.thread.daemon = True
        self.thread.start()
        self.command_phrases = ["what is happening", "describe", "what's in front of me", "tell me about this",
                               "what do you see", "can you describe", "please describe"]

    def _speech_worker(self):
        with self.mic as source:
            self.recognizer.adjust_for_ambient_noise(source, duration=2)
        while self.running:
            if self.is_listening:
                try:
                    with self.mic as source:
                        audio = self.recognizer.listen(source, timeout=None, phrase_time_limit=5)
                    text = self.recognizer.recognize_google(audio).lower()
                    with self.lock:
                        self.current_speech_text = text
                    if any(cmd in text for cmd in self.command_phrases):
                        if self.speech_queue.empty():
                            self.speech_queue.put_nowait(text)
                except sr.UnknownValueError:
                    with self.lock:
                        self.current_speech_text = "Sorry, I didn't catch that. Please try again."
                except sr.RequestError:
                    with self.lock:
                        self.current_speech_text = "Speech service unavailable. Please check your internet connection."
                except Exception as e:
                    logging.error(f"Speech error: {e}")
                    with self.lock:
                        self.current_speech_text = "There was an error processing your request."
            time.sleep(0.1)

    def get_speech_text(self):
        with self.lock:
            return self.current_speech_text

    def check_for_command(self):
        if not self.speech_queue.empty():
            return self.speech_queue.get_nowait()
        return None

    def pause_listening(self):
        self.is_listening = False

    def resume_listening(self):
        self.is_listening = True

    def stop(self):
        self.running = False
        self.thread.join()

# Frame Storage and Retrieval
class FrameStorage:
    def __init__(self, db_path, csv_file):
        self.db_path = db_path
        self.csv_file = csv_file
        self.lock = Lock()
        self.running = True
        self.frame_queue = Queue(maxsize=10)  # Limit queue size to avoid memory issues
        self.store_thread = Thread(target=self._store_worker)
        self.store_thread.daemon = True
        self.store_thread.start()

    def _store_worker(self):
        # Create a new connection in this thread
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        while self.running:
            if not self.frame_queue.empty():
                frame, timestamp = self.frame_queue.get()
                try:
                    # Convert frame to binary
                    _, buffer = cv2.imencode('.jpg', frame)
                    frame_binary = buffer.tobytes()
                    # Store in SQLite
                    c = conn.cursor()
                    c.execute("INSERT INTO frames (timestamp, frame) VALUES (?, ?)",
                              (timestamp, frame_binary))
                    frame_id = c.lastrowid
                    conn.commit()
                    # Log to CSV
                    with self.lock:
                        with open(self.csv_file, 'a', newline='') as f:
                            writer = csv.writer(f)
                            writer.writerow([frame_id, timestamp])
                    logger.info(f"Stored frame {frame_id} at {timestamp}")
                except Exception as e:
                    logger.error(f"Error storing frame: {e}")
            time.sleep(0.05)
        conn.close()

    def store_frame(self, frame, timestamp):
        if self.frame_queue.full():
            self.frame_queue.get()  # Remove oldest frame if queue is full
        self.frame_queue.put((frame, timestamp))

    def get_latest_frame(self):
        # Create a new connection in this thread
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
            logger.error(f"Error retrieving frame: {e}")
            return None
        finally:
            conn.close()

    def stop(self):
        self.running = False
        self.store_thread.join()

# Caption Generator using stored frames
class CaptionGenerator:
    def __init__(self, processor, model, device, speech_engine, frame_storage):
        self.processor = processor
        self.model = model
        self.device = device
        self.speech_engine = speech_engine
        self.frame_storage = frame_storage
        self.current_caption = "Visual description system ready"
        self.last_spoken_caption = ""
        self.lock = Lock()
        self.running = True
        self.thread = Thread(target=self._caption_worker)
        self.thread.daemon = True
        self.thread.start()

    def _caption_worker(self):
        while self.running:
            try:
                frame = self.frame_storage.get_latest_frame()
                if frame is not None:
                    caption = self._generate_caption(frame)
                    with self.lock:
                        self.current_caption = caption
                else:
                    with self.lock:
                        self.current_caption = "No recent frames available to describe."
            except Exception as e:
                logging.error(f"Caption error: {e}")
                with self.lock:
                    self.current_caption = "Having trouble seeing right now. Please try again."
            time.sleep(0.5)  # Process every 0.5 seconds to reduce load

    def _generate_caption(self, image):
        try:
            image_resized = cv2.resize(image, (224, 224))
            rgb_image = cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB)
            inputs = self.processor(images=rgb_image, return_tensors="pt")
            inputs = {name: tensor.to(self.device).to(torch.float16) if self.device == 'cuda' else tensor.to(self.device)
                     for name, tensor in inputs.items()}
            with torch.no_grad():
                outputs = self.model.generate(**inputs, max_length=50)
            caption = self.processor.batch_decode(outputs, skip_special_tokens=True)[0]
            return caption
        except Exception as e:
            logging.error(f"Caption generation failed: {e}")
            return "I'm having trouble generating a description right now."

    def get_caption(self):
        with self.lock:
            return self.current_caption

    def speak_description(self, user_question=None):
        with self.lock:
            if user_question:
                self.speech_engine.say(f"You asked: {user_question}")
            if self.current_caption:
                if self.current_caption != self.last_spoken_caption:
                    self.speech_engine.say("Here's what I see: " + self.current_caption)
                    self.last_spoken_caption = self.current_caption
                else:
                    self.speech_engine.say("The scene hasn't changed since last time.")
            else:
                self.speech_engine.say("I am unable to generate a description at the moment. Please try again later.")
            self.speech_engine.runAndWait()

    def stop(self):
        self.running = False
        self.thread.join()

# Gesture Detector
class GestureDetector:
    def __init__(self):
        self.current_gesture = "No gesture detected"
        self.landmarks_result = None
        self.gesture_queue = Queue(maxsize=1)
        self.lock = Lock()
        self.running = True
        self.thread = Thread(target=self._gesture_worker)
        self.thread.daemon = True
        self.thread.start()

    def _gesture_worker(self):
        while self.running:
            try:
                if not self.gesture_queue.empty():
                    frame = self.gesture_queue.get()
                    gesture, result = self._detect_gesture(frame)
                    with self.lock:
                        self.current_gesture = gesture
                        self.landmarks_result = result
            except Exception as e:
                logging.error(f"Gesture detection error: {str(e)}")
                with self.lock:
                    self.current_gesture = "Gesture detection failed"
                    self.landmarks_result = None
            time.sleep(0.05)

    def _detect_gesture(self, frame):
        framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(framergb)

        if result.multi_hand_landmarks:
            landmarks = []
            for handslms in result.multi_hand_landmarks:
                for lm in handslms.landmark:
                    lmx = lm.x  # Normalized [0, 1]
                    lmy = lm.y
                    landmarks.append([lmx, lmy])

                landmarks = np.array(landmarks, dtype=np.float32)
                mean_x = np.mean(landmarks[:, 0])
                mean_y = np.mean(landmarks[:, 1])
                landmarks[:, 0] -= mean_x
                landmarks[:, 1] -= mean_y
                landmarks_array = landmarks.flatten()
                landmarks_array = np.expand_dims(landmarks_array, axis=0)

                landmarks_tensor = tf.convert_to_tensor(landmarks_array, dtype=tf.float32)
                prediction_dict = gesture_model(landmarks_tensor)
                prediction_key = list(prediction_dict.keys())[0]
                prediction = prediction_dict[prediction_key]
                prediction_np = prediction.numpy()
                classID = np.argmax(prediction_np)
                return f"Gesture: {gesture_classNames[classID]}", result
        return "No gesture detected", result

    def update_frame(self, frame):
        if self.gesture_queue.empty():
            try:
                self.gesture_queue.put_nowait(frame.copy())
            except:
                pass

    def get_gesture(self):
        with self.lock:
            return self.current_gesture

    def get_landmarks_result(self):
        with self.lock:
            return self.landmarks_result

    def stop(self):
        self.running = False
        self.thread.join()

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

def speak_welcome_message(engine):
    welcome_msg = """Welcome to the Visual Assistant. 
    I can help you understand your surroundings. Use voice commands like 'what is happening' to hear a description. 
    Gestures will be detected and displayed on the screen."""
    engine.say(welcome_msg)
    engine.runAndWait()

def main():
    processor, model, device = load_models()
    if None in (processor, model):
        return

    # Initialize speech engine
    engine = pyttsx3.init()
    engine.setProperty('rate', 145)
    engine.setProperty('volume', 1.0)
    voices = engine.getProperty('voices')
    engine.setProperty('voice', voices[1].id if len(voices) > 1 else voices[0].id)

    # Initialize database and CSV
    db_path = 'frames.db'
    conn = init_database()  # Create database in main thread
    csv_file = init_csv()
    frame_storage = FrameStorage(db_path, csv_file)

    # Initialize components
    speech_handler = SpeechHandler()
    caption_gen = CaptionGenerator(processor, model, device, engine, frame_storage)
    gesture_detector = GestureDetector()

    # Speak welcome message
    speak_welcome_message(engine)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        engine.say("I'm having trouble accessing the camera. Please check if it's properly connected.")
        engine.runAndWait()
        return

    is_assistance_enabled = True
    frame_count = 0
    prev_time = time.time()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                engine.say("I'm having trouble accessing the camera. Please check if it's properly connected.")
                engine.runAndWait()
                break

            frame_count += 1

            # Store frame every 5 frames to reduce database load
            if frame_count % 5 == 0:
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                frame_storage.store_frame(frame, timestamp)

            # Update gesture detection every 3 frames
            if frame_count % 3 == 0:
                gesture_detector.update_frame(frame)

            # Draw hand landmarks
            result = gesture_detector.get_landmarks_result()
            if result and result.multi_hand_landmarks:
                for handslms in result.multi_hand_landmarks:
                    mpDraw.draw_landmarks(frame, handslms, mpHands.HAND_CONNECTIONS)

            # Get detected gesture (for display only)
            current_gesture = gesture_detector.get_gesture()

            # Process caption and speech if enabled
            if is_assistance_enabled:
                speech_handler.resume_listening()

                # Check for voice commands
                user_question = speech_handler.check_for_command()
                if user_question:
                    speech_handler.pause_listening()
                    caption_gen.speak_description(user_question)
                    speech_handler.resume_listening()
            else:
                speech_handler.pause_listening()

            # Get captions for display
            blip_caption = caption_gen.get_caption()
            speech_text = speech_handler.get_speech_text()

            # Calculate FPS
            curr_time = time.time()
            fps = 1 / (curr_time - prev_time)
            prev_time = curr_time

            # Display information
            y_offset = 40
            cv2.putText(frame, f"Gesture: {current_gesture}", (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            y_offset += 30
            cv2.putText(frame, f"Status: {'Enabled' if is_assistance_enabled else 'Disabled'}", (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            y_offset += 30
            cv2.putText(frame, f"Description: {blip_caption}", (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            y_offset += 30
            cv2.putText(frame, f"Speech: {speech_text}", (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            y_offset += 30
            cv2.putText(frame, get_gpu_usage(), (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            y_offset += 30
            cv2.putText(frame, f"FPS: {fps:.2f}", (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            cv2.imshow("Visual Assistant", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                engine.say("Goodbye. Shutting down the visual assistant.")
                engine.runAndWait()
                break

    finally:
        caption_gen.stop()
        gesture_detector.stop()
        speech_handler.stop()
        frame_storage.stop()
        cap.release()
        cv2.destroyAllWindows()
        conn.close()

if __name__ == "__main__":
    main()