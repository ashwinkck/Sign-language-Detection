import os
import cv2
import numpy as np
import tensorflow as tf
from unidecode import unidecode
from cvzone.HandTrackingModule import HandDetector
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
import pyttsx3
import threading
import time

class SignLanguageDetector:
    def __init__(self):
        # Constants
        self.IMG_SIZE = 128
        self.MODEL_PATH = r'C:\Users\ashwi\OneDrive\Desktop\Sycamore\SLD mark1.5\sign_language_model.h5'
        self.DATA_DIR = r"C:\Users\ashwi\OneDrive\Desktop\Sycamore\SLD mark1.5\Data"
        self.CONFIDENCE_THRESHOLD = 70

        # Initialize components
        self.verify_paths()
        self.init_camera()
        self.init_model()
        self.init_tts_engine()

    def verify_paths(self):
        """Verify that all required paths exist."""
        if not os.path.exists(self.MODEL_PATH):
            raise FileNotFoundError(f"Model not found at: {self.MODEL_PATH}")
        if not os.path.exists(self.DATA_DIR):
            raise FileNotFoundError(f"Data directory not found at: {self.DATA_DIR}")

    def init_camera(self):
        """Initialize the camera with fallback options."""
        for port in range(4):
            self.cap = cv2.VideoCapture(port)
            if self.cap.isOpened():
                ret, frame = self.cap.read()
                if ret and frame is not None:
                    print(f"Camera initialized on port {port}")
                    self.detector = HandDetector(maxHands=1)
                    return
                self.cap.release()
        raise RuntimeError("No working camera found")

    def init_model(self):
        """Initialize the TensorFlow model and load class names."""
        try:
            self.model = tf.keras.models.load_model(self.MODEL_PATH)
            self.class_names = [unidecode(name.strip()) for name in os.listdir(self.DATA_DIR)]
            print(f"Loaded {len(self.class_names)} classes")
        except Exception as e:
            raise RuntimeError(f"Failed to load model: {str(e)}")

    def init_tts_engine(self):
        """Initialize text-to-speech engine."""
        try:
            self.engine = pyttsx3.init()
            voices = self.engine.getProperty('voices')
            self.engine.setProperty('voice', voices[1].id)  # Change index for different voices
            self.engine.setProperty('rate', 150)
        except Exception as e:
            print(f"TTS initialization failed: {str(e)}")
            self.engine = None

class SignLanguageApp(tk.Tk):
    def __init__(self):
        super().__init__()

        try:
            self.detector = SignLanguageDetector()
            self.setup_ui()
            self.current_gesture = None
            self.current_confidence = None
            self.sentence = ""
            self.is_running = True
            self.setup_bindings()
        except Exception as e:
            messagebox.showerror("Initialization Error", str(e))
            self.destroy()
            return

    def setup_ui(self):
        """Setup the user interface."""
        self.title("Sign Language Detection")
        self.setup_video_frame()
        self.setup_prediction_table()
        self.setup_sentence_display()

    def setup_video_frame(self):
        """Setup the video display frame."""
        self.video_frame = ttk.Frame(self)
        self.video_frame.pack(pady=10)
        self.canvas = tk.Canvas(self.video_frame, width=640, height=480)
        self.canvas.pack()

    def setup_prediction_table(self):
        """Setup the prediction display table."""
        self.prediction_frame = ttk.Frame(self)
        self.prediction_frame.pack(pady=10)
        self.table = ttk.Treeview(
            self.prediction_frame, 
            columns=("Gesture", "Confidence"), 
            show='headings'
        )
        self.table.heading("Gesture", text="Gesture Detected")
        self.table.heading("Confidence", text="Confidence Level (%)")
        self.table.pack()

    def setup_sentence_display(self):
        """Setup the sentence display label.""" 
        self.sentence_label = ttk.Label(self, text="", font=("Arial", 16))
        self.sentence_label.pack(pady=10)

    def setup_bindings(self):
        """Setup key bindings and window protocols."""
        self.bind('<i>', self.save_gesture)
        self.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.after(30, self.update_video)

    def update_video(self):
        """Update video frame and process for gesture detection."""
        if not self.is_running:
            return

        try:
            ret, frame = self.detector.cap.read()
            if ret:
                frame = cv2.flip(frame, 1)
                self.process_frame(frame)
            self.after(30, self.update_video)
        except Exception as e:
            messagebox.showerror("Video Error", f"Error processing video: {str(e)}")
            self.on_closing()

    def process_frame(self, frame):
        """Process video frame for gesture detection."""
        hands, frame = self.detector.detector.findHands(frame)
        
        if hands:
            hand = hands[0]
            x, y, w, h = hand['bbox']
            padding = 20
            x, y = max(0, x - padding), max(0, y - padding)
            w = min(frame.shape[1] - x, w + 2 * padding)
            h = min(frame.shape[0] - y, h + 2 * padding)
            
            crop = frame[y:y + h, x:x + w]
            if crop.size > 0:
                self.process_hand(crop, frame)

        self.update_display(frame)

    def process_hand(self, crop, frame):
        """Process detected hand for gesture recognition."""
        img_batch = self.preprocess_image(crop)
        predictions = self.detector.model.predict(img_batch, verbose=0)
        
        class_id = np.argmax(predictions[0])
        confidence = predictions[0][class_id] * 100

        if confidence > self.detector.CONFIDENCE_THRESHOLD:
            gesture = self.detector.class_names[class_id]
            self.update_current_prediction(gesture, confidence, frame)

    def preprocess_image(self, frame):
        """Preprocess image for model prediction."""
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (self.detector.IMG_SIZE, self.detector.IMG_SIZE))
        normalized = resized / 255.0
        return np.expand_dims(normalized, axis=0)

    def update_current_prediction(self, gesture, confidence, frame):
        """Update current prediction and display."""
        self.current_gesture = gesture
        self.current_confidence = confidence
        
        # Update UI elements
        self.display_gesture_on_frame(frame, gesture, confidence)
        self.update_table(gesture, confidence)

        # Trigger text-to-speech in a separate thread
        if self.detector.engine:
            threading.Thread(target=self.speak_gesture, args=(gesture,)).start()

    def display_gesture_on_frame(self, frame, gesture, confidence):
        """Display gesture information on video frame."""
        cv2.putText(frame, f'Gesture: {gesture}', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f'Confidence: {confidence:.2f}%', (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    def update_display(self, frame):
        """Update the video display."""
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        img_tk = ImageTk.PhotoImage(image=img)
        self.canvas.create_image(0, 0, anchor='nw', image=img_tk)
        self.canvas.image = img_tk

    def update_table(self, gesture, confidence):
        """Update prediction table with current results."""
        self.table.delete(*self.table.get_children())
        self.table.insert("", "end", values=(gesture, f"{confidence:.2f}"))

    def speak_gesture(self, gesture):
        """Speak the detected gesture."""
        try:
            self.detector.engine.say(gesture)
            self.detector.engine.runAndWait()
        except Exception as e:
            print(f"TTS error: {str(e)}")

    def save_gesture(self, event):
        """Save current gesture to the sentence."""
        if self.current_gesture and self.current_confidence:
            # Check for space gesture (assuming 'space' is one of your gestures)
            if self.current_gesture.lower() == "space":
                self.sentence += " "  # Add space for word separation
            else:
                self.sentence += self.current_gesture + " "

            self.sentence_label.config(text=self.sentence.strip())
            self.table.insert("", "end", values=(
                self.current_gesture,
                f"{self.current_confidence:.2f}"
            ))

    def on_closing(self):
        """Clean up resources on application close."""
        self.is_running = False
        if hasattr(self, 'detector'):
            self.detector.cap.release()
        self.destroy()

if __name__ == "__main__":
    try:
        app = SignLanguageApp()
        app.mainloop()
    except Exception as e:
        print(f"Application Error: {str(e)}")
