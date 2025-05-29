"""
Real-time ASL-letter detector
——————————————
 • Requires:  ultralytics, opencv-python, tkinter, pillow
 • Usage:    python asl_webcam.py
"""

from ultralytics import YOLO
import cv2, time, os
import tkinter as tk
from asl_gui import ASLDetectorGUI
from datetime import datetime

class ASLDetectorApp:
    def __init__(self):
        # --- configuration ---
        self.WEIGHTS = "weights/internet_model_v1.pt"  # Path to the trained YOLO model
        self.CONF_TH = 0.83  # Confidence threshold for detection (83%)
        self.FONT = cv2.FONT_HERSHEY_SIMPLEX  # Font for text display
        self.COL_LABEL = (0, 255, 0)  # Green color for labels (BGR format)
        self.COL_BOX = (0, 255, 255)  # Yellow color for bounding boxes (BGR format)
        
        # FPS calculation variables
        self.frame_count = 0
        self.fps = 0
        self.fps_start_time = time.time()
        self.fps_update_interval = 1.0  # Update FPS every second
        
        # Word spelling configuration
        self.HOLD_TIME = 1.0  # Time in seconds to hold a sign before adding to word
        self.current_word = ""  # Current word being spelled
        self.last_detected_letter = None  # Last letter detected by the model
        self.letter_hold_start = None  # Time when current letter detection started
        self.word_history = []  # List to store history of spelled words
        
        # Create directory for saving spelled words
        os.makedirs("words", exist_ok=True)
        
        # Initialize variables
        self.cap = None  # Video capture object
        self.model = YOLO(self.WEIGHTS)  # Load the YOLO model
        self.is_running = False  # Flag to track if detection is running
        
        # Create root window and GUI
        self.root = tk.Tk()
        self.gui = ASLDetectorGUI(
            self.root,
            on_webcam_change=self.on_webcam_change,
            on_toggle_detection=self.toggle_detection,
            on_clear_word=self.clear_word,
            on_submit_word=self.submit_word,
            on_clear_history=self.clear_history,
            on_backspace=self.backspace_word,
            on_confidence_change=self.on_confidence_change,
            on_space=self.add_space
        )
        
        # Initialize available webcams
        self.initialize_webcams()
    
    def initialize_webcams(self):
        """Find and list all available webcams in the system"""
        available_webcams = []
        for i in range(10):  # Check first 10 indices for webcams
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                available_webcams.append(f"Webcam {i}")
                cap.release()
        self.gui.set_webcam_list(available_webcams)
    
    def on_webcam_change(self, webcam_index):
        """Handle webcam selection change from GUI"""
        if self.is_running:
            self.stop_detection()
        
        self.cap = cv2.VideoCapture(webcam_index)
        if not self.cap.isOpened():
            self.gui.update_status(f"Error: Could not open webcam {webcam_index}")
            return
        self.gui.update_status(f"Webcam {webcam_index} selected")
    
    def toggle_detection(self):
        """Toggle detection on/off based on current state"""
        if self.is_running:
            self.stop_detection()
        else:
            self.start_detection()
    
    def start_detection(self):
        """Start the ASL detection process"""
        if not self.cap:
            webcam_index = self.gui.get_selected_webcam()
            if webcam_index is None:
                self.gui.update_status("Please select a webcam first")
                return
            self.cap = cv2.VideoCapture(webcam_index)
            if not self.cap.isOpened():
                self.gui.update_status(f"Error: Could not open webcam {webcam_index}")
                return
        
        self.is_running = True
        self.gui.update_start_button(True)
        self.gui.update_status("Detection running - Press 's' to save frame, Esc to quit")
        self.update_frame()
    
    def stop_detection(self):
        """Stop the ASL detection process and release resources"""
        self.is_running = False
        self.gui.update_start_button(False)
        if self.cap:
            self.cap.release()
            self.cap = None
        self.gui.update_status("Detection stopped")
    
    def clear_word(self):
        """Clear the current word and reset letter detection state"""
        self.current_word = ""
        self.last_detected_letter = None
        self.letter_hold_start = None
        self.gui.update_word(self.current_word)
        self.gui.update_status("Word cleared")
    
    def backspace_word(self):
        """Remove the last letter from the current word"""
        if self.current_word:
            self.current_word = self.current_word[:-1]
            self.gui.update_word(self.current_word)
            self.gui.update_status("Removed last letter")
        else:
            self.gui.update_status("No letter to remove")
    
    def clear_history(self):
        """Clear the word history and the saved file"""
        self.word_history = []
        self.gui.update_word_history(self.word_history)
        # Clear the file
        open("words/spelled_words.txt", "w").close()
        self.gui.update_status("Word history cleared")
    
    def submit_word(self):
        """Submit the current word to history and save to file"""
        if self.current_word:
            # Add to history
            self.word_history.append(self.current_word)
            
            # Save to file with timestamp
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            with open("words/spelled_words.txt", "a", encoding="utf-8") as f:
                f.write(f"{timestamp}: {self.current_word}\n")
            
            # Update GUI
            self.gui.update_status(f"Submitted word: {self.current_word}")
            self.gui.update_word_history(self.word_history)
            
            # Clear current word
            self.clear_word()
        else:
            self.gui.update_status("No word to submit")
    
    def handle_letter_detection(self, letter, score):
        """Handle letter detection for word spelling with hold time"""
        current_time = time.time()
        
        if letter != self.last_detected_letter:
            # New letter detected, start hold timer
            self.last_detected_letter = letter
            self.letter_hold_start = current_time
            self.gui.update_status(f"Detected {letter} - Hold to add to word")
        elif self.letter_hold_start and (current_time - self.letter_hold_start) >= self.HOLD_TIME:
            # Letter held long enough, add to word
            self.current_word += letter
            self.gui.update_word(self.current_word)
            self.gui.update_status(f"Added {letter} to word: {self.current_word}")
            self.letter_hold_start = None  # Reset hold timer
    
    def update_frame(self):
        """Update the video frame with detection results"""
        if not self.is_running:
            return
            
        # Read frame from webcam
        ok, frame = self.cap.read()
        if not ok:
            self.gui.update_status("Error: Could not read frame")
            self.stop_detection()
            return
            
        # Run YOLO model inference on the frame
        results = self.model(frame, conf=self.CONF_TH, verbose=False)[0]
        
        # Process each detected sign
        for box in results.boxes:
            # Get detection details
            cls = int(box.cls[0])  # Class index of detected letter
            score = float(box.conf[0])  # Confidence score
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box coordinates
            
            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), self.COL_BOX, 2)
            
            # Get letter name and create label
            letter = self.model.names[cls]
            label = f"{letter} {score*100:0.1f}%"
            
            # Draw label above box
            cv2.putText(frame, label, (x1, y1-8), self.FONT, 0.7, self.COL_LABEL, 2,
                        cv2.LINE_AA)
            
            # Handle letter detection for word spelling
            self.handle_letter_detection(letter, score)
        
        # Calculate and display FPS
        self.frame_count += 1
        current_time = time.time()
        elapsed_time = current_time - self.fps_start_time
        
        if elapsed_time >= self.fps_update_interval:
            self.fps = self.frame_count / elapsed_time
            self.frame_count = 0
            self.fps_start_time = current_time
        
        # Draw FPS on frame
        fps_text = f"FPS: {self.fps:.1f}"
        cv2.putText(frame, fps_text, (10, 30), self.FONT, 1, (0, 255, 0), 2, cv2.LINE_AA)
        
        # Update GUI with processed frame
        self.gui.update_video_frame(frame)
        
        # Schedule next frame update
        self.root.after(10, self.update_frame)
    
    def on_confidence_change(self, value):
        """Handle confidence threshold changes from GUI"""
        self.CONF_TH = value
        self.gui.update_status(f"Confidence threshold set to {value:.2f}")
    
    def add_space(self):
        """Add a space to the current word"""
        self.current_word += " "
        self.gui.update_word(self.current_word)
        self.gui.update_status("Added space")
    
    def run(self):
        """Start the application main loop"""
        self.root.mainloop()

def main():
    """Main entry point of the application"""
    app = ASLDetectorApp()
    app.run()

if __name__ == "__main__":
    main()
