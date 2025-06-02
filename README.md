# ASL Letter Detector

A real-time American Sign Language (ASL) letter detection application that uses computer vision to recognize hand signs and allows users to spell words through sign language.

## Features

- Real-time ASL letter detection using YOLO object detection
- Word spelling functionality by holding signs
- Adjustable confidence threshold for detection accuracy
- Multiple webcam support
- Word history tracking
- Simple and intuitive GUI interface

## Requirements

- Python
- OpenCV (opencv-python)
- Ultralytics YOLO
- Tkinter (usually comes with Python)
- Pillow (PIL)

## Installation

1. Clone this repository:
```bash
git clone https://github.com/Sign-Language-Detection/New-Sign-Language-Detection
cd New-Sign-Language-Detection
```

2. Install the required packages:
```bash
pip install ultralytics opencv-python pillow
```

## Usage

1. Run the application

2. Using the application:
   - Select your webcam from the dropdown menu
   - Click "Start" to begin detection
   - Hold your hand sign steady to add a letter to the current word
   - Use the "Back" button to remove the last letter
   - Use the "Clear" button to start a new word
   - Click "Submit" to save the current word
   - Adjust the confidence slider to fine-tune detection accuracy
   - Click "Stop" to pause detection

## Word History

- The application keeps track of your spelled words
- Words are saved to `words/spelled_words.txt` with timestamps
- Use the "Clear History" button to reset the word history

## Project Structure

```
.
├── asl_webcam.py      # Main application file
├── asl_gui.py         # GUI implementation
├── weights/           # Directory for YOLO weights
└── words/             # Directory for word history
    └── spelled_words.txt
```

## Notes

- The application uses a confidence threshold of 0.83 by default
- You can adjust the confidence threshold using the slider in the GUI
- The application requires good lighting conditions for optimal detection
- Make sure your hand signs are clearly visible to the camera
