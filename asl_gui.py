"""
ASL Detector GUI Interface
——————————————
 • Contains the GUI components for the ASL detector application
"""

import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import cv2

class ASLDetectorGUI:
    def __init__(self, root, on_webcam_change, on_toggle_detection, on_clear_word, on_submit_word, on_clear_history, on_backspace, on_confidence_change, on_space):
        self.root = root
        self.root.title("ASL Detector")
        
        # Prevent window resizing
        self.root.resizable(False, False)
        
        # Store callback functions
        self.on_webcam_change = on_webcam_change
        self.on_toggle_detection = on_toggle_detection
        self.on_clear_word = on_clear_word
        self.on_submit_word = on_submit_word
        self.on_clear_history = on_clear_history
        self.on_backspace = on_backspace
        self.on_confidence_change = on_confidence_change
        self.on_space = on_space
        
        # Initialize variables
        self.video_frame = None
        self.status_var = None
        self.webcam_var = None
        self.start_button = None
        self.word_var = None
        self.history_text = None
        self.confidence_var = None
        
        # Create main container
        self.main_container = ttk.Frame(self.root)
        self.main_container.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Create GUI elements
        self.create_widgets()
    
    def create_widgets(self):
        # Control Frame
        control_frame = ttk.Frame(self.main_container)
        control_frame.pack(fill=tk.X, pady=(0, 5))
        
        # Webcam selection
        ttk.Label(control_frame, text="Select Webcam:").pack(side=tk.LEFT, padx=5)
        self.webcam_var = tk.StringVar()
        self.webcam_combo = ttk.Combobox(control_frame, textvariable=self.webcam_var, width=15)
        self.webcam_combo.pack(side=tk.LEFT, padx=5)
        self.webcam_combo.bind('<<ComboboxSelected>>', self._on_webcam_change)
        
        # Start/Stop button
        self.start_button = ttk.Button(control_frame, text="Start", command=self._on_toggle_detection)
        self.start_button.pack(side=tk.LEFT, padx=5)
        
        # Confidence threshold slider
        ttk.Label(control_frame, text="Confidence:").pack(side=tk.LEFT, padx=5)
        self.confidence_var = tk.DoubleVar(value=0.83)
        confidence_slider = ttk.Scale(
            control_frame,
            from_=0.01,
            to=1.00,
            orient=tk.HORIZONTAL,
            variable=self.confidence_var,
            length=150,
            command=self._on_confidence_change
        )
        confidence_slider.pack(side=tk.LEFT, padx=5)
        
        # Confidence value label
        self.confidence_label = ttk.Label(control_frame, text="0.83")
        self.confidence_label.pack(side=tk.LEFT, padx=5)
        
        # Word Frame
        word_frame = ttk.LabelFrame(self.main_container, text="Spelling", padding="5")
        word_frame.pack(fill=tk.X, pady=(0, 5))
        
        # Current word display
        self.word_var = tk.StringVar()
        word_label = ttk.Label(word_frame, textvariable=self.word_var, font=('Arial', 24))
        word_label.pack(side=tk.LEFT, padx=5)
        
        # Word control buttons
        back_button = ttk.Button(word_frame, text="Back", command=self._on_backspace)
        back_button.pack(side=tk.LEFT, padx=5)
        
        space_button = ttk.Button(word_frame, text="Space", command=self._on_space)
        space_button.pack(side=tk.LEFT, padx=5)
        
        clear_button = ttk.Button(word_frame, text="Clear", command=self._on_clear_word)
        clear_button.pack(side=tk.LEFT, padx=5)
        
        submit_button = ttk.Button(word_frame, text="Submit", command=self._on_submit_word)
        submit_button.pack(side=tk.LEFT, padx=5)
        
        # History Frame
        history_frame = ttk.LabelFrame(self.main_container, text="Word History", padding="5")
        history_frame.pack(fill=tk.X, pady=(0, 5))
        
        # Add a horizontal frame for the button just inside the LabelFrame
        history_top = ttk.Frame(history_frame)
        history_top.pack(fill=tk.X, pady=(0, 2))
        clear_history_button = ttk.Button(history_top, text="Clear History", command=self._on_clear_history)
        clear_history_button.pack(side=tk.RIGHT, padx=2)
        
        # History text widget
        self.history_text = tk.Text(history_frame, height=3, width=30, wrap=tk.WORD)
        self.history_text.pack(fill=tk.X, padx=5, pady=(0,5))
        self.history_text.config(state=tk.DISABLED)  # Make read-only
        
        # Video Frame
        self.video_frame = ttk.Label(self.main_container)
        self.video_frame.pack(fill=tk.BOTH, expand=True)
        
        # Status bar
        self.status_var = tk.StringVar()
        self.status_var.set("Ready")
        status_bar = ttk.Label(self.main_container, textvariable=self.status_var, relief=tk.SUNKEN)
        status_bar.pack(fill=tk.X, pady=(5, 0))
    
    def _on_webcam_change(self, event=None):
        webcam_index = int(self.webcam_var.get().split()[-1])
        self.on_webcam_change(webcam_index)
    
    def _on_toggle_detection(self):
        self.on_toggle_detection()
    
    def _on_clear_word(self):
        self.on_clear_word()
    
    def _on_submit_word(self):
        self.on_submit_word()
    
    def _on_clear_history(self):
        self.on_clear_history()
    
    def _on_backspace(self):
        self.on_backspace()
    
    def _on_confidence_change(self, *args):
        """Handle confidence threshold changes"""
        value = self.confidence_var.get()
        self.confidence_label.config(text=f"{value:.2f}")
        self.on_confidence_change(value)
    
    def _on_space(self):
        """Handle space button click"""
        self.on_space()
    
    def update_video_frame(self, frame):
        """Update the video frame with a new image"""
        # Resize frame to 640x480 before displaying
        frame_resized = cv2.resize(frame, (640, 480))
        frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb)
        imgtk = ImageTk.PhotoImage(image=img)
        self.video_frame.imgtk = imgtk
        self.video_frame.configure(image=imgtk)
        
        # Update window size based on frame size
        if not self.root.winfo_ismapped():
            self.root.update_idletasks()
            width = img.width + 20  # Add padding
            height = img.height + 250  # Add space for controls and status bar
            self.root.geometry(f"{width}x{height}")
    
    def update_status(self, message):
        """Update the status bar message"""
        self.status_var.set(message)
    
    def update_word(self, word):
        """Update the current word display"""
        self.word_var.set(word)
    
    def update_word_history(self, history):
        """Update the word history display"""
        self.history_text.config(state=tk.NORMAL)  # Enable editing
        self.history_text.delete(1.0, tk.END)  # Clear current content
        
        # Show last 3 words (or fewer if less available)
        for word in history[-3:]:
            self.history_text.insert(tk.END, word + "\n")
        
        self.history_text.config(state=tk.DISABLED)  # Make read-only again
    
    def set_webcam_list(self, webcams):
        """Set the list of available webcams in the dropdown"""
        self.webcam_combo['values'] = webcams
    
    def update_start_button(self, is_running):
        """Update the start/stop button text"""
        self.start_button.configure(text="Stop" if is_running else "Start")
    
    def get_selected_webcam(self):
        """Get the currently selected webcam index"""
        if not self.webcam_var.get():
            return None
        return int(self.webcam_var.get().split()[-1]) 