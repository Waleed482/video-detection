

import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from tensorflow.keras.layers import TFSMLayer
import torch
import time
import sys
import pyttsx3
from threading import Thread, Lock
from queue import Queue
from datetime import datetime
from utils import init_database, init_csv, init_description_csv, get_gpu_usage, load_models, logger
from speech_handler import SpeechHandler
from storage import FrameStorage, DescriptionStorage
import sqlite3
from dateDescription import listen_for_date, parse_spoken_date, fetch_descriptions_for_date

# Initialize MediaPipe for gesture recognition
mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mpDraw = mp.solutions.drawing_utils

# Load gesture recognition model
gesture_model = TFSMLayer('mp_hand_gesture', call_endpoint='serving_default')
with open('gestures.names', 'r') as f:
    gesture_classNames = f.read().split('\n')
logger.info(f"Gesture class names: {gesture_classNames}")

class SpeechQueue:
    def __init__(self, engine):
        self.engine = engine
        self.queue = Queue()
        self.running = True
        self.thread = Thread(target=self._process_queue)
        self.thread.daemon = True
        self.thread.start()

    def _process_queue(self):
        while self.running:
            if not self.queue.empty():
                text = self.queue.get()
                self.engine.say(text)
                self.engine.runAndWait()
            time.sleep(0.1)

    def add_speech(self, text):
        self.queue.put(text)

    def stop(self):
        self.running = False
        self.thread.join()

# Caption Generator using stored frames
class CaptionGenerator:
    def __init__(self, processor, model, device, speech_queue, frame_storage, description_storage):
        self.processor = processor
        self.model = model
        self.device = device
        self.speech_queue = speech_queue
        self.frame_storage = frame_storage
        self.description_storage = description_storage
        self.current_caption = "Visual description system ready"
        self.last_spoken_caption = ""
        self.last_stored_time = time.time()
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
                        current_time = time.time()
                        if current_time - self.last_stored_time >= 5:
                            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                            self.description_storage.store_description(caption, timestamp)
                            self.last_stored_time = current_time
                else:
                    with self.lock:
                        self.current_caption = "No recent frames available to describe."
            except Exception as e:
                logger.error(f"Caption error: {e}")
                with self.lock:
                    self.current_caption = "Having trouble seeing right now. Please try again."
            time.sleep(0.5)

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
            logger.error(f"Caption generation failed: {e}")
            return "I'm having trouble generating a description right now."

    def get_caption(self):
        with self.lock:
            return self.current_caption

    def speak_description(self, user_question=None):
        with self.lock:
            messages = []
            if user_question:
                messages.append(f"You asked: {user_question}")
            if self.current_caption:
                if self.current_caption != self.last_spoken_caption:
                    messages.append("Here's what I see: " + self.current_caption)
                    self.last_spoken_caption = self.current_caption
                else:
                    messages.append("The scene hasn't changed since last time.")
            else:
                messages.append("I am unable to generate a description at the moment. Please try again later.")
            for msg in messages:
                self.speech_queue.add_speech(msg)

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
                logger.error(f"Gesture detection error: {str(e)}")
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
                    lmx = lm.x
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

def speak_welcome_message(speech_queue):
    welcome_msg = """Welcome to the Visual Assistant. """
    speech_queue.add_speech(welcome_msg)


def main():
    processor, model, device = load_models()
    if None in (processor, model):
        return

    # Initialize speech engine and queue
    engine = pyttsx3.init()
    engine.setProperty('rate', 145)
    engine.setProperty('volume', 1.0)
    voices = engine.getProperty('voices')
    engine.setProperty('voice', voices[1].id if len(voices) > 1 else voices[0].id)
    speech_queue = SpeechQueue(engine)

    # Initialize database and CSV for frames and descriptions
    db_path = 'frames.db'
    conn = init_database()
    csv_file = init_csv()
    description_csv_file = init_description_csv()
    frame_storage = FrameStorage(db_path, csv_file)
    description_storage = DescriptionStorage(db_path, description_csv_file)

    # Initialize components
    speech_handler = SpeechHandler()
    caption_gen = CaptionGenerator(processor, model, device, speech_queue, frame_storage, description_storage)
    gesture_detector = GestureDetector()

    # Speak welcome message
    speak_welcome_message(speech_queue)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        speech_queue.add_speech("I'm having trouble accessing the camera. Please check if it's properly connected.")
        return

    is_assistance_enabled = True
    frame_count = 0
    prev_time = time.time()
    last_report_time = 0
    report_cooldown = 5  # Seconds between report triggers
    last_stop_time = 0
    stop_cooldown = 5
    waiting_for_date = False
    waiting_for_time = False
    date_query_timeout = 10  # Seconds to wait for date/time input
    date_query_start_time = None
    user_provided_date = None

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                speech_queue.add_speech("I'm having trouble accessing the camera. Please check if it's properly connected.")
                break

            frame_count += 1

            # Store frame every 5 frames to reduce database load
            # if frame_count % 5 == 0:
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

            # Process gesture recognition
            current_gesture = gesture_detector.get_gesture()

            # Handle stop gesture for report
            # if "stop" in current_gesture.lower():
            #     current_time = time.time()
            #     if current_time - last_report_time >= report_cooldown:
            #         latest_description, timestamp = description_storage.get_latest_description()
            #         if latest_description:
            #             report_msg = f"Here's the latest report from {timestamp}: {latest_description}"
            #             speech_queue.add_speech(report_msg)
            #         else:
            #             speech_queue.add_speech("No previous descriptions are available.")
            #         last_report_time = current_time
            #     # Resume SpeechHandler listening if it was paused
            #     speech_handler.resume_listening()


            if "rock" in current_gesture.lower():
                current_time = time.time()
                if current_time - last_stop_time >= stop_cooldown:
                    if not waiting_for_date:
                        # Pause SpeechHandler listening to focus on date input
                        speech_handler.pause_listening()
                        # Start listening for the date
                        waiting_for_date = True
                        date_query_start_time = current_time
                        print("Detected stop gesture. Waiting for date input...")
                    last_stop_time = current_time

            # Process date input if waiting
            if waiting_for_date:
                # Initialize user_provided_date to None at the start of the block
                user_provided_date = None
                # Listen for the spoken date using SpeechHandler
                spoken_text = listen_for_date(speech_queue, speech_handler)
                if spoken_text:
                    print(f"You said: {spoken_text}")
                    user_provided_date = parse_spoken_date(spoken_text, speech_queue)
                    if user_provided_date:
                        print(f"Parsed date: {user_provided_date}")
                        fetch_descriptions_for_date(user_provided_date, speech_queue, db_path, conn)
                        waiting_for_date = False  # Reset after processing
                    else:
                        waiting_for_date = False  # Reset on failure
                elif current_time - date_query_start_time >= date_query_timeout:
                    speech_queue.add_speech("I didn't hear a date. Please try the stop gesture again.")
                    print("No date heard within timeout.")
                    waiting_for_date = False
                    date_query_start_time = None
                # Resume SpeechHandler listening after date query (success or timeout)
                if not waiting_for_date:
                    speech_handler.resume_listening()

            # Process caption and speech if enabled
            if is_assistance_enabled:
                speech_handler.resume_listening()

                # Check for description commands
                user_question = speech_handler.check_for_command()
                if user_question:
                    speech_handler.pause_listening()
                    caption_gen.speak_description(user_question)
                    speech_handler.resume_listening()

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
                speech_queue.add_speech("Goodbye. Shutting down the visual assistant.")
                break

    finally:
        print("Cleaning up resources...")
        
        # Ensure all processes are stopped gracefully
        if 'speech_queue' in globals():
            speech_queue.stop()
        if 'caption_gen' in globals():
            caption_gen.stop()
        if 'gesture_detector' in globals():
            gesture_detector.stop()
        if 'speech_handler' in globals():
            speech_handler.stop()
        if 'frame_storage' in globals():
            frame_storage.stop()
        if 'description_storage' in globals():
            description_storage.stop()

        # Release video capture properly
        if 'cap' in locals() and cap.isOpened():
            cap.release()

        cv2.destroyAllWindows()

        # Close database connection safely
        if 'conn' in globals():
            try:
                conn.close()
            except Exception as e:
                print(f"Error closing database connection: {e}")

        print("Visual Assistant has been successfully shut down.")

if __name__ == "__main__":
    main()