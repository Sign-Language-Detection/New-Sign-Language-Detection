"""
Real-time ASL-letter detector
——————————————
 • Requires:  ultralytics, opencv-python, tkinter, pillow
 • Usage:    python asl_webcam.py
 • Keys:     Esc - quit
"""

from ultralytics import YOLO
import cv2, time, os
import tkinter as tk
from asl_gui import ASLDetectorGUI
from datetime import datetime

class ASLDetectorApp:
    def __init__(self):
        # --- configuration ---
        self.WEIGHTS = "weights/internet_data.pt"
        self.CONF_TH = 0.83
        self.FONT = cv2.FONT_HERSHEY_SIMPLEX
        self.COL_LABEL = (0, 255, 0)
        self.COL_BOX = (0, 255, 255)
        
        # Word spelling configuration
        self.HOLD_TIME = 1.0  # seconds to hold a sign before adding to word
        self.current_word = ""
        self.last_detected_letter = None
        self.letter_hold_start = None
        self.word_history = []  # Store history of spelled words
        
        # Create directories
        os.makedirs("words", exist_ok=True)
        
        # Initialize variables
        self.cap = None
        self.model = YOLO(self.WEIGHTS)
        self.is_running = False
        
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
            on_confidence_change=self.on_confidence_change
        )
        
        # Initialize available webcams
        self.initialize_webcams()
    
    def initialize_webcams(self):
        """Find and list all available webcams"""
        available_webcams = []
        for i in range(10):  # Check first 10 indices
            cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
            if cap.isOpened():
                available_webcams.append(f"Webcam {i}")
                cap.release()
        self.gui.set_webcam_list(available_webcams)
    
    def on_webcam_change(self, webcam_index):
        """Handle webcam selection change"""
        if self.is_running:
            self.stop_detection()
        
        self.cap = cv2.VideoCapture(webcam_index, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            self.gui.update_status(f"Error: Could not open webcam {webcam_index}")
            return
        self.gui.update_status(f"Webcam {webcam_index} selected")
    
    def toggle_detection(self):
        """Toggle detection on/off"""
        if self.is_running:
            self.stop_detection()
        else:
            self.start_detection()
    
    def start_detection(self):
        """Start the detection process"""
        if not self.cap:
            webcam_index = self.gui.get_selected_webcam()
            if webcam_index is None:
                self.gui.update_status("Please select a webcam first")
                return
            self.cap = cv2.VideoCapture(webcam_index, cv2.CAP_DSHOW)
            if not self.cap.isOpened():
                self.gui.update_status(f"Error: Could not open webcam {webcam_index}")
                return
        
        self.is_running = True
        self.gui.update_start_button(True)
        self.gui.update_status("Detection running - Press 's' to save frame, Esc to quit")
        self.update_frame()
    
    def stop_detection(self):
        """Stop the detection process"""
        self.is_running = False
        self.gui.update_start_button(False)
        if self.cap:
            self.cap.release()
            self.cap = None
        self.gui.update_status("Detection stopped")
    
    def clear_word(self):
        """Clear the current word"""
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
        """Submit the current word"""
        if self.current_word:
            # Add to history
            self.word_history.append(self.current_word)
            
            # Save to file
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
        """Handle letter detection for word spelling"""
        current_time = time.time()
        
        if letter != self.last_detected_letter:
            self.last_detected_letter = letter
            self.letter_hold_start = current_time
            self.gui.update_status(f"Detected {letter} - Hold to add to word")
        elif self.letter_hold_start and (current_time - self.letter_hold_start) >= self.HOLD_TIME:
            # Add letter to word if held long enough
            self.current_word += letter
            self.gui.update_word(self.current_word)
            self.gui.update_status(f"Added {letter} to word: {self.current_word}")
            self.letter_hold_start = None  # Reset hold timer
    
    def update_frame(self):
        """Update the video frame with detection results"""
        if not self.is_running:
            return
            
        ok, frame = self.cap.read()
        if not ok:
            self.gui.update_status("Error: Could not read frame")
            self.stop_detection()
            return
            
        # Run inference
        results = self.model(frame, conf=self.CONF_TH, verbose=False)[0]
        
        # Draw boxes and labels
        for box in results.boxes:
            cls = int(box.cls[0])
            score = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(frame, (x1, y1), (x2, y2), self.COL_BOX, 2)
            letter = self.model.names[cls]
            label = f"{letter} {score*100:0.1f}%"
            cv2.putText(frame, label, (x1, y1-8), self.FONT, 0.7, self.COL_LABEL, 2,
                        cv2.LINE_AA)
            
            # Handle letter detection for word spelling
            self.handle_letter_detection(letter, score)
        
        # Update GUI
        self.gui.update_video_frame(frame)
        
        # Check for key presses
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # Esc
            self.stop_detection()
            return
        
        # Schedule next frame update
        self.root.after(10, self.update_frame)
    
    def on_confidence_change(self, value):
        """Handle confidence threshold changes"""
        self.CONF_TH = value
        self.gui.update_status(f"Confidence threshold set to {value:.2f}")
    
    def run(self):
        """Start the application"""
        self.root.mainloop()

def main():
    app = ASLDetectorApp()
    app.run()

if __name__ == "__main__":
    main()
