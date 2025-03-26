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

# Initialize MediaPipe for gesture recognition
mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mpDraw = mp.solutions.drawing_utils

# Load gesture recognition model
gesture_model = TFSMLayer('mp_hand_gesture', call_endpoint='serving_default')
with open('gestures.names', 'r') as f:
    gesture_classNames = f.read().split('\n')

# Setup logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

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
        self.command_phrases = ["what is happening","describe", "what's in front of me", "tell me about this", 
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

class CaptionGenerator:
    def __init__(self, processor, model, device, speech_engine):
        self.processor = processor
        self.model = model
        self.device = device
        self.speech_engine = speech_engine
        self.current_caption = "Visual description system ready"
        self.last_spoken_caption = ""
        self.caption_queue = Queue(maxsize=1)
        self.lock = Lock()
        self.running = True
        self.thread = Thread(target=self._caption_worker)
        self.thread.daemon = True
        self.thread.start()

    def _caption_worker(self):
        while self.running:
            try:
                if not self.caption_queue.empty():
                    frame = self.caption_queue.get()
                    caption = self._generate_caption(frame)
                    with self.lock:
                        self.current_caption = caption
            except Exception as e:
                logging.error(f"Caption error: {e}")
                with self.lock:
                    self.current_caption = "Having trouble seeing right now. Please try again."
            time.sleep(0.05)

    def _generate_caption(self, image):
        try:
            image_resized = cv2.resize(image, (224, 224))
            rgb_image = cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB)
            inputs = self.processor(images=rgb_image, return_tensors="pt")
            inputs = {name: tensor.to(self.device) for name, tensor in inputs.items()}
            with torch.no_grad():
                outputs = self.model.generate(**inputs, max_length=50)
            caption = self.processor.batch_decode(outputs, skip_special_tokens=True)[0]
            return caption
        except Exception as e:
            logging.error(f"Caption generation failed: {e}")
            return "I'm having trouble generating a description right now."

    def update_frame(self, frame):
        if self.caption_queue.empty():
            try:
                self.caption_queue.put_nowait(frame.copy())
            except:
                pass

    def get_caption(self):
        with self.lock:
            return self.current_caption

    def speak_description(self, user_question=None):
        with self.lock:
            # Speak the user's question if provided
            if user_question:
                self.speech_engine.say(f"You asked: {user_question}")
            
            # Speak the current description if it's new
            if self.current_caption != self.last_spoken_caption:
                self.speech_engine.say("Here's what I see: " + self.current_caption)
                self.last_spoken_caption = self.current_caption
            else:
                self.speech_engine.say("The scene hasn't changed since last time")
            
            self.speech_engine.runAndWait()

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
    I can help you understand your surroundings."""
    engine.say(welcome_msg)
    engine.runAndWait()

def main():
    processor, model, device = load_models()
    if None in (processor, model):
        return

    # Initialize speech engine with clearer settings
    engine = pyttsx3.init()
    engine.setProperty('rate', 145)  # Slightly slower for better comprehension
    engine.setProperty('volume', 1.0)
    voices = engine.getProperty('voices')
    engine.setProperty('voice', voices[1].id)  # Typically a female voice
    
    # Speak welcome message
    speak_welcome_message(engine)
    
    cap = cv2.VideoCapture(0)
    speech_handler = SpeechHandler()
    caption_gen = CaptionGenerator(processor, model, device, engine)
    
    is_assistance_enabled = True
    is_video_paused = False  # Track video state
    prev_gesture = ""
    last_status_change = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            engine.say("I'm having trouble accessing the camera. Please check if it's properly connected.")
            engine.runAndWait()
            break

        # Process gesture recognition
        framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        gesture_result = hands.process(framergb)
        current_gesture = ""
        
        if gesture_result.multi_hand_landmarks:
            for hand_landmarks in gesture_result.multi_hand_landmarks:
                # Process landmarks
                landmarks = []
                for lm in hand_landmarks.landmark:
                    landmarks.extend([lm.x * frame.shape[1], lm.y * frame.shape[0]])
                
                # Predict gesture
                prediction = gesture_model(np.array([landmarks], dtype=np.float32))
                class_id = np.argmax(prediction['dense_16'].numpy())
                current_gesture = gesture_classNames[class_id]
                
                # Detect fist gesture and speak description
                if current_gesture == 'fist' and prev_gesture != 'fist':
                    caption_gen.speak_description()
                
                # Toggle video pause/play on 'stop' gesture
                if current_gesture == 'stop' and prev_gesture != 'stop' and time.time() - last_status_change > 3:
                    is_video_paused = not is_video_paused
                    status = "paused" if is_video_paused else "resumed"
                    engine.say(f"Video {status}")
                    engine.runAndWait()
                    last_status_change = time.time()
                
                mpDraw.draw_landmarks(frame, hand_landmarks, mpHands.HAND_CONNECTIONS)
        
        prev_gesture = current_gesture
        
        # Process caption and speech if enabled
        if is_assistance_enabled:
            caption_gen.update_frame(frame)
            speech_handler.resume_listening()
            
            # Check for voice commands
            user_question = speech_handler.check_for_command()
            if user_question:
                # Ensure we have the latest description
                caption_gen.update_frame(frame)
                time.sleep(0.1)  # Brief pause to allow caption update
                caption_gen.speak_description(user_question)  # Speak the description when the user asks for it
        else:
            speech_handler.pause_listening()

        # Get captions for display
        blip_caption = caption_gen.get_caption()
        speech_text = speech_handler.get_speech_text()
        
        # Display information (for sighted assistants)
        y_offset = 40
        cv2.putText(frame, f"Gesture: {current_gesture}", (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
        y_offset += 30
        cv2.putText(frame, f"Status: {'Enabled' if is_assistance_enabled else 'Disabled'}", (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
        y_offset += 30
        cv2.putText(frame, f"Description: {blip_caption}", (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
        y_offset += 30
        cv2.putText(frame, f"Speech: {speech_text}", (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
        y_offset += 30
        cv2.putText(frame, get_gpu_usage(), (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
        
        cv2.imshow("Visual Assistant", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            engine.say("Goodbye. Shutting down the visual assistant.")
            engine.runAndWait()
            break

    caption_gen.stop()
    speech_handler.stop()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
