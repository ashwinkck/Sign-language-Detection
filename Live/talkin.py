import os
import cv2
import numpy as np
import tensorflow as tf
from unidecode import unidecode
from cvzone.HandTrackingModule import HandDetector
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import pyttsx3
import threading
import time  # Import time module for sleep functionality

# Constants
IMG_SIZE = 128
MODEL_PATH = r'C:\Users\ashwi\OneDrive\Desktop\Sycamore\SLD mark1.5\sign_language_model.h5'
DATA_DIR = r"C:\Users\ashwi\OneDrive\Desktop\Sycamore\SLD mark1.5\Data"
CONFIDENCE_THRESHOLD = 70
HOLD_TIME = 2  # Duration for speech output in seconds

# Tkinter UI setup
class SignLanguageApp:
    def __init__(self, master):
        self.master = master
        self.master.title("Sign Language Detection")

        # Initialize text-to-speech engine
        self.engine = pyttsx3.init()
        voices = self.engine.getProperty('voices')
        self.engine.setProperty('voice', voices[1].id)  # Select the second voice, typically feminine
        self.engine.setProperty('rate', 150)  # Adjust speech rate

        # Video frame
        self.video_frame = ttk.Frame(master)
        self.video_frame.pack(pady=10)

        self.canvas = tk.Canvas(self.video_frame, width=640, height=480)
        self.canvas.pack()

        # Prediction table
        self.prediction_frame = ttk.Frame(master)
        self.prediction_frame.pack(pady=10)

        self.table = ttk.Treeview(self.prediction_frame, columns=("Gesture", "Confidence"), show='headings')
        self.table.heading("Gesture", text="Gesture Detected")
        self.table.heading("Confidence", text="Confidence Level (%)")
        self.table.pack()

        # Initialize camera and start detection
        self.cap, self.detector, self.model, self.class_names = self.initialize()
        self.current_gesture = None
        self.current_confidence = None

        # Frame count for prediction control
        self.frame_count = 0

        # Bind "i" key to save the gesture
        self.master.bind('<i>', self.save_gesture)

        self.update_video()

    def initialize(self):
        """Load model, class names, and initialize camera."""
        model = tf.keras.models.load_model(MODEL_PATH)
        class_names = os.listdir(DATA_DIR)
        class_names = [unidecode(name.strip()) for name in class_names]

        cap = cv2.VideoCapture(0)  # Adjust camera port as needed
        detector = HandDetector(maxHands=1)

        return cap, detector, model, class_names

    def update_video(self):
        """Capture video frame and process for prediction."""
        ret, frame = self.cap.read()
        if ret:
            # Process the frame in a separate thread
            threading.Thread(target=self.process_frame, args=(frame,)).start()

        # Call this method again after 30ms
        self.master.after(30, self.update_video)

    def process_frame(self, frame):
        """Process the frame for gesture prediction."""
        frame = cv2.flip(frame, 1)  # Mirror the frame
        hands, frame = self.detector.findHands(frame)

        if hands:
            hand = hands[0]
            x, y, w, h = hand['bbox']
            padding = 20
            x = max(0, x - padding)
            y = max(0, y - padding)
            w = min(frame.shape[1] - x, w + 2 * padding)
            h = min(frame.shape[0] - y, h + 2 * padding)

            img_crop = frame[y:y + h, x:x + w]
            if img_crop.size > 0:
                img_batch = self.preprocess_image(img_crop)
                predictions = self.model.predict(img_batch, verbose=0)

                class_id = np.argmax(predictions[0])
                confidence = predictions[0][class_id] * 100

                if confidence > CONFIDENCE_THRESHOLD and 0 <= class_id < len(self.class_names):
                    class_label = self.class_names[class_id]
                    self.current_gesture = class_label
                    self.current_confidence = confidence

                    # Display gesture and confidence on the video feed
                    self.display_gesture_on_screen(frame, class_label, confidence)

                    # Update table with current gesture and confidence (live prediction)
                    self.update_table(class_label, confidence)

                    # Read the gesture using text-to-speech in a separate thread
                    threading.Thread(target=self.read_gesture, args=(class_label,)).start()

        # Convert the frame to PhotoImage and display on canvas
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        img_tk = ImageTk.PhotoImage(image=img)
        self.canvas.create_image(0, 0, anchor='nw', image=img_tk)
        self.canvas.image = img_tk  # Keep a reference to avoid garbage collection

    def preprocess_image(self, frame):
        """Preprocess the image for model prediction."""
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, (IMG_SIZE, IMG_SIZE))
        img_normalized = img_resized / 255.0
        img_batch = np.expand_dims(img_normalized, axis=0)
        return img_batch

    def update_table(self, gesture, confidence):
        """Update the prediction table with the latest results (live prediction)."""
        # Clear previous entry and display the current gesture and confidence live
        self.table.delete(*self.table.get_children())  # Clear previous entries
        self.table.insert("", "end", values=(gesture, f"{confidence:.2f}"))

    def display_gesture_on_screen(self, frame, gesture, confidence):
        """Display the detected gesture and confidence on the video frame."""
        cv2.putText(frame, f'Gesture: {gesture}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(frame, f'Confidence: {confidence:.2f}%', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)

    def read_gesture(self, gesture):
        """Use text-to-speech to read the detected gesture in a separate thread."""
        self.engine.say(gesture)
        self.engine.runAndWait()  # Block until the speech is finished
        time.sleep(HOLD_TIME)  # Keep the thread alive for HOLD_TIME seconds

    def save_gesture(self, event):
        """Save the current gesture and confidence in the table when 'i' key is pressed."""
        if self.current_gesture and self.current_confidence:
            self.table.insert("", "end", values=(self.current_gesture, f"{self.current_confidence:.2f}"))
            print(f"Gesture Saved to UI: {self.current_gesture}, Confidence: {self.current_confidence:.2f}")

    def on_closing(self):
        """Handle window closing."""
        self.cap.release()
        self.master.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = SignLanguageApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)  # Handle closing the app
    root.mainloop()
