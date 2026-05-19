# Gesture Canvas — CV for Academics 🎓
Computer Vision Based Interactive Teaching System

[![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)](https://www.python.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.13-green?logo=opencv)](https://opencv.org/)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10.9-orange)](https://developers.google.com/mediapipe)

---

## 📌 Project Overview
A fully gesture-controlled interactive whiteboard and 3D canvas built entirely with **Python, OpenCV, and MediaPipe** — no hardware beyond a webcam required.

The system lets teachers and students draw, annotate, spawn 2D/3D shapes, and interact with the board using only their hands and voice — making it a zero-cost interactive teaching tool suitable for any classroom.

**Real-world use cases:**
- Interactive whiteboard replacement for classrooms
- Live 3D shape visualization for geometry and physics
- Gesture-controlled presentations at exhibitions
- Low-cost smart board for schools without expensive hardware

---

## 🎯 What's New (v3)
- 🎙️ **Voice commands** — change color, mode, spawn shapes, clear, OCR all by voice
- 🗣️ **Say shape names** — say "cube", "sphere", "triangle" etc to spawn instantly
- 🔢 **Multiple shapes** — unlimited shapes on screen simultaneously, each independently grabbable
- 🔍 **Smooth zoom** — pinch two hands to scale shapes smoothly with interpolation
- 📝 **EasyOCR** — write with your finger, say "read" → text recognized and overlaid
- 📐 **Auto shape snapping** — draw rough shapes, say "snap" → perfect geometry
- 🎨 **Recolor while holding** — grab a shape and say a color to recolor it live
- 🔄 **Persistent canvas** — whiteboard and shapes survive mode switches
- 🗑️ **Selective clear** — clear text only, shapes only, or everything separately

---

## 🧠 Technologies Used
| Library | Purpose |
|---|---|
| **Python 3.11** | Core language |
| **OpenCV** | Camera feed, drawing, rendering |
| **MediaPipe** | Real-time hand tracking, finger detection |
| **NumPy** | 3D math, projections, rotations |
| **SpeechRecognition** | Voice command listener |
| **PyAudio** | Microphone access |
| **EasyOCR** | Handwriting recognition |
| **PyTorch** | Backend for EasyOCR |

---

## ⚙️ Features

### Whiteboard Mode
- Draw with 1–3 fingers
- Erase with 4 fingers
- Auto snap drawn shape to perfect geometry with 5 fingers or say "snap"
- Say "read" to recognize handwritten text via OCR

### 2D Shapes Mode
- Spawn rectangle, circle, triangle, star, pentagon
- Both hands pinch → spawn default shape
- Left pinch + right fingers (2–5) → spawn specific shape
- Grab and move any shape independently
- Two-hand pinch spread/close → smooth zoom in/out

### 3D Shapes Mode
- Spawn cube, pyramid, cylinder, cone, sphere
- All shapes rotate automatically
- Full 3D projection with lighting and face shading
- Same grab, move, zoom gestures as 2D

### Voice Commands
| Say | Action |
|---|---|
| Shape name ("cube", "circle" etc) | Spawn that shape |
| Color name ("red", "blue" etc) | Change color |
| "whiteboard" / "2d" / "3d" | Switch mode |
| "clear" / "clear text" | Clear canvas only |
| "clear shapes" | Remove shapes only |
| "clear all" | Wipe everything |
| "snap" | Snap drawn shape |
| "read" / "ocr" | Read handwriting |
| "delete" | Delete grabbed shape |

### Keyboard Shortcuts
| Key | Action |
|---|---|
| C | Clear canvas |
| X | Clear shapes |
| Z | Clear everything |
| S | Snap shape |
| O | OCR read |
| ESC | Quit |

---

## 🚀 Getting Started

### Installation
pip install opencv-python mediapipe==0.10.9 numpy
pip install SpeechRecognition pyaudio
pip install torch torchvision
pip install easyocr

### Run
python main.py

---

## 📂 Project Structure
main.py              # Main application
requirements.txt     # All dependencies
README.md            # This file
.gitignore           # Ignores venv, cache, etc.

---

## 🗺️ Roadmap
- [ ] Session save and export to PDF/image
- [ ] Multi-user collaborative whiteboard over network
- [ ] Student attention tracking via face mesh
- [ ] Quiz mode with gesture-based answers
- [ ] Integration with object detection robot (in progress)

---

## 👨‍💻 Author
**Sulav Sangroula**
Student | Computer Vision Enthusiast
GitHub: https://github.com/Sulav16

---

## 📜 License
This project is for educational purposes only.
You may use it to learn, demonstrate, and improve your skills,
but do not use for commercial purposes without permission.
