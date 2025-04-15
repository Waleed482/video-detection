import speech_recognition as sr
from threading import Thread, Lock
from queue import Queue
import time
import logging

class SpeechHandler:
    def __init__(self):
        self.recognizer = sr.Recognizer()
        self.mic = sr.Microphone()
        self.current_speech_text = "System ready. Waiting for your command..."
        self.lock = Lock()
        self.running = True
        self.is_listening = True
        self.speech_queue = Queue(maxsize=1) # For general commands
        self.date_response_queue = Queue(maxsize=1) # For date input
        self.thread = Thread(target=self._speech_worker)
        self.thread.daemon = True
        self.thread.start()

        self.recognizer.pause_threshold = 1.0
        self.recognizer.phrase_threshold = 0.5
        self.recognizer.non_speaking_duration = 0.5

        self.command_phrases = ["what is happening", "describe", "what's in front of me",
                                "tell me about this", "what do you see", "can you describe"]

    def _speech_worker(self):
        with self.mic as source:
            self.recognizer.adjust_for_ambient_noise(source, duration=2)

        while self.running:
            if self.is_listening:
                try:
                    with self.mic as source:
                        audio = self.recognizer.listen(
                            source,
                            timeout=5,
                            phrase_time_limit=10
                        )
                    text = self.recognizer.recognize_google(audio).lower()

                    with self.lock:
                        self.current_speech_text = text

                    if any(cmd in text for cmd in self.command_phrases):
                        if self.speech_queue.empty():
                            self.speech_queue.put_nowait(text)

                except sr.WaitTimeoutError:
                    continue
                except sr.UnknownValueError:
                    with self.lock:
                        self.current_speech_text = "Sorry, I didn't catch that."
                except sr.RequestError:
                    with self.lock:
                        self.current_speech_text = "Speech service unavailable."
                except Exception as e:
                    logging.error(f"Speech error: {e}")
                    with self.lock:
                        self.current_speech_text = "A speech processing error occurred."
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

