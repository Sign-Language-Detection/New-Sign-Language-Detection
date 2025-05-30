"""
Real-time ASL-letter detector
——————————————
 • Requires:  ultralytics, opencv-python, tkinter, pillow
 • Usage:    python asl_webcam.py
"""

# Import necessary libraries
from ultralytics import YOLO  # For AI model that detects ASL signs
import cv2, time, os  # OpenCV for video, time for delays, os for file operations
import tkinter as tk  # For creating the graphical user interface
from asl_gui import ASLDetectorGUI  # Custom GUI class for our application
from datetime import datetime  # For timestamping saved words

class ASLDetectorApp:
    def __init__(self):
        # --- Configuration Settings ---
        self.WEIGHTS = "weights/mixed_v3.pt"  # Path to the trained YOLO model that recognizes ASL signs
        self.CONF_TH = 0.83  # Confidence threshold (83%) - only show detections with 83% or higher confidence
        self.FONT = cv2.FONT_HERSHEY_SIMPLEX  # Font style for text display on video
        self.COL_LABEL = (0, 255, 0)  # Green color for labels (BGR format)
        self.COL_BOX = (0, 255, 255)  # Yellow color for bounding boxes (BGR format)
        
        # --- Word Spelling Configuration ---
        self.HOLD_TIME = 1.0  # Time in seconds to hold a sign before adding to word (prevents accidental additions)
        self.current_word = ""  # Stores the word being currently spelled
        self.last_detected_letter = None  # Tracks the last letter detected by the model
        self.letter_hold_start = None  # Records when the current letter detection started
        self.word_history = []  # List to store history of previously spelled words
        
        # Create a directory to save spelled words if it doesn't exist
        os.makedirs("words", exist_ok=True)
        
        # --- Initialize Core Components ---
        self.cap = None  # Video capture object (will hold the webcam feed)
        self.model = YOLO(self.WEIGHTS)  # Load the YOLO model for ASL detection
        self.is_running = False  # Flag to track if detection is currently running
        
        # Create the main window and GUI
        self.root = tk.Tk()
        self.gui = ASLDetectorGUI(
            self.root,
            on_webcam_change=self.on_webcam_change,  # Function to handle webcam selection changes
            on_toggle_detection=self.toggle_detection,  # Function to start/stop detection
            on_clear_word=self.clear_word,  # Function to clear current word
            on_submit_word=self.submit_word,  # Function to save current word
            on_clear_history=self.clear_history,  # Function to clear word history
            on_backspace=self.backspace_word,  # Function to remove last letter
            on_confidence_change=self.on_confidence_change,  # Function to adjust detection confidence
            on_space=self.add_space  # Function to add space to word
        )
        
        # Find and list all available webcams
        self.initialize_webcams()
    
    def initialize_webcams(self):
        """Find and list all available webcams in the system"""
        available_webcams = []
        # Check first 10 possible webcam indices
        for i in range(10):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():  # If webcam is accessible
                available_webcams.append(f"Webcam {i}")
                cap.release()  # Release the webcam after checking
        self.gui.set_webcam_list(available_webcams)  # Update GUI with found webcams
    
    def on_webcam_change(self, webcam_index):
        """Handle webcam selection change from GUI"""
        # If detection is running, stop it before changing webcam
        if self.is_running:
            self.stop_detection()
        
        # Try to open the selected webcam
        self.cap = cv2.VideoCapture(webcam_index)
        if not self.cap.isOpened():
            self.gui.update_status(f"Error: Could not open webcam {webcam_index}")
            return
        self.gui.update_status(f"Webcam {webcam_index} selected")
    
    def toggle_detection(self):
        """Toggle detection on/off based on current state"""
        if self.is_running:
            self.stop_detection()  # If running, stop it
        else:
            self.start_detection()  # If stopped, start it
    
    def start_detection(self):
        """Start the ASL detection process"""
        # If no webcam is currently open, try to open the selected one
        if not self.cap:
            webcam_index = self.gui.get_selected_webcam()
            if webcam_index is None:
                self.gui.update_status("Please select a webcam first")
                return
            self.cap = cv2.VideoCapture(webcam_index)
            if not self.cap.isOpened():
                self.gui.update_status(f"Error: Could not open webcam {webcam_index}")
                return
        
        # Start the detection process
        self.is_running = True
        self.gui.update_start_button(True)  # Update button to show "Stop"
        self.gui.update_status("Detection running - Press 's' to save frame, Esc to quit")
        self.update_frame()  # Start processing frames
    
    def stop_detection(self):
        """Stop the ASL detection process and release resources"""
        self.is_running = False
        self.gui.update_start_button(False)  # Update button to show "Start"
        if self.cap:
            self.cap.release()  # Release the webcam
            self.cap = None
        self.gui.update_status("Detection stopped")
    
    def clear_word(self):
        """Clear the current word and reset letter detection state"""
        self.current_word = ""  # Clear the current word
        self.last_detected_letter = None  # Reset last detected letter
        self.letter_hold_start = None  # Reset hold timer
        self.gui.update_word(self.current_word)  # Update GUI
        self.gui.update_status("Word cleared")
    
    def backspace_word(self):
        """Remove the last letter from the current word"""
        if self.current_word:
            self.current_word = self.current_word[:-1]  # Remove the last character
            self.gui.update_word(self.current_word)  # Update the word display
            self.gui.update_status("Removed last letter")
        else:
            self.gui.update_status("No letter to remove")
    
    def clear_history(self):
        """Clear the word history and the saved file"""
        self.word_history = []  # Clear the history list
        self.gui.update_word_history(self.word_history)  # Update the history display
        # Clear the file by opening it in write mode
        open("words/spelled_words.txt", "w").close()
        self.gui.update_status("Word history cleared")
    
    def submit_word(self):
        """Submit the current word to history and save to file"""
        if self.current_word:
            # Add the word to history list
            self.word_history.append(self.current_word)
            
            # Save to file with current timestamp
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            with open("words/spelled_words.txt", "a", encoding="utf-8") as f:
                f.write(f"{timestamp}: {self.current_word}\n")
            
            # Update the GUI
            self.gui.update_status(f"Submitted word: {self.current_word}")
            self.gui.update_word_history(self.word_history)
            
            # Clear the current word for next input
            self.clear_word()
        else:
            self.gui.update_status("No word to submit")
    
    def handle_letter_detection(self, letter, score):
        """Handle letter detection for word spelling with hold time"""
        current_time = time.time()
        
        if letter != self.last_detected_letter:
            # New letter detected, start the hold timer
            self.last_detected_letter = letter
            self.letter_hold_start = current_time
            self.gui.update_status(f"Detected {letter} - Hold to add to word")
        elif self.letter_hold_start and (current_time - self.letter_hold_start) >= self.HOLD_TIME:
            # Letter has been held long enough, add it to the word
            self.current_word += letter
            self.gui.update_word(self.current_word)
            self.gui.update_status(f"Added {letter} to word: {self.current_word}")
            self.letter_hold_start = None  # Reset the hold timer
    
    def update_frame(self):
        """Update the video frame with detection results"""
        if not self.is_running:
            return
            
        # Read a frame from the webcam
        ok, frame = self.cap.read()
        if not ok:
            self.gui.update_status("Error: Could not read frame")
            self.stop_detection()
            return
            
        # Run the YOLO model to detect ASL signs in the frame
        results = self.model(frame, conf=self.CONF_TH, verbose=False)[0]
        
        # Process each detected sign
        for box in results.boxes:
            # Get detection details
            cls = int(box.cls[0])  # Class index of detected letter
            score = float(box.conf[0])  # Confidence score
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box coordinates
            
            # Draw yellow bounding box around the detected sign
            cv2.rectangle(frame, (x1, y1), (x2, y2), self.COL_BOX, 2)
            
            # Get letter name and create label with confidence percentage
            letter = self.model.names[cls]
            label = f"{letter} {score*100:0.1f}%"
            
            # Draw green label above the box
            cv2.putText(frame, label, (x1, y1-8), self.FONT, 0.7, self.COL_LABEL, 2,
                        cv2.LINE_AA)
            
            # Handle letter detection for word spelling
            self.handle_letter_detection(letter, score)
        
        # Update the GUI with the processed frame
        self.gui.update_video_frame(frame)
        
        # Schedule the next frame update (creates a continuous loop)
        self.root.after(10, self.update_frame)
    
    def on_confidence_change(self, value):
        """Handle confidence threshold changes from GUI"""
        self.CONF_TH = value  # Update the confidence threshold
        self.gui.update_status(f"Confidence threshold set to {value:.2f}")
    
    def add_space(self):
        """Add a space to the current word"""
        self.current_word += " "  # Add a space to the current word
        self.gui.update_word(self.current_word)  # Update the word display
        self.gui.update_status("Added space")
    
    def run(self):
        """Start the application main loop"""
        self.root.mainloop()  # Start the Tkinter event loop

def main():
    """Main entry point of the application"""
    app = ASLDetectorApp()  # Create the application instance
    app.run()  # Start the application

if __name__ == "__main__":
    main()  # Run the application when script is executed directly
