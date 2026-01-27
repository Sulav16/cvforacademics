import cv2
import numpy as np
import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# -----------------------
# Load Mediapipe Hand Landmarker (Tasks API)
# -----------------------
MODEL_PATH = "hand_landmarker.task"  # make sure this file is in the same folder

base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=1,
    running_mode=vision.RunningMode.VIDEO  # <-- VIDEO mode for real-time
)
landmarker = vision.HandLandmarker.create_from_options(options)

# -----------------------
# Cube drawing function
# -----------------------
def draw_cube():
    glBegin(GL_QUADS)

    # Top face
    glColor3f(1, 0, 0)
    glVertex3f(1, 1, -1)
    glVertex3f(-1, 1, -1)
    glVertex3f(-1, 1, 1)
    glVertex3f(1, 1, 1)

    # Bottom face
    glColor3f(0, 1, 0)
    glVertex3f(1, -1, 1)
    glVertex3f(-1, -1, 1)
    glVertex3f(-1, -1, -1)
    glVertex3f(1, -1, -1)

    # Front face
    glColor3f(0, 0, 1)
    glVertex3f(1, 1, 1)
    glVertex3f(-1, 1, 1)
    glVertex3f(-1, -1, 1)
    glVertex3f(1, -1, 1)

    # Back face
    glColor3f(1, 1, 0)
    glVertex3f(1, -1, -1)
    glVertex3f(-1, -1, -1)
    glVertex3f(-1, 1, -1)
    glVertex3f(1, 1, -1)

    glEnd()

# -----------------------
# Pygame + OpenGL setup
# -----------------------
pygame.init()
display = (800, 600)
pygame.display.set_mode(display, DOUBLEBUF | OPENGL)

gluPerspective(45, display[0] / display[1], 0.1, 50.0)
glEnable(GL_DEPTH_TEST)

rot_x, rot_y = 0, 0
zoom = -7

# -----------------------
# OpenCV capture
# -----------------------
cap = cv2.VideoCapture(0)
frame_counter = 0

# -----------------------
# Main loop
# -----------------------
running = True

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    success, frame = cap.read()
    if not success:
        continue

    frame = cv2.flip(frame, 1)  # Mirror the frame

    # Convert frame to Mediapipe Image
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)

    # Run detection for VIDEO mode
    result = landmarker.detect_for_video(mp_image, frame_counter)
    frame_counter += 1

    # If landmarks detected
    if result.hand_landmarks:
        hand = result.hand_landmarks[0]

        # Index finger tip for rotation
        index_tip = hand[8]
        rot_y = (index_tip.x - 0.5) * 180
        rot_x = (index_tip.y - 0.5) * 180

        # Pinch gesture for zoom
        thumb_tip = hand[4]
        dist = np.linalg.norm(np.array([thumb_tip.x, thumb_tip.y]) -
                              np.array([index_tip.x, index_tip.y]))
        if dist < 0.05:
            zoom += 0.1
        else:
            zoom -= 0.01

        # Draw landmarks on frame
        for lm in hand:
            x, y = int(lm.x * frame.shape[1]), int(lm.y * frame.shape[0])
            cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

    # -----------------------
    # Render cube
    # -----------------------
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    glPushMatrix()

    glTranslatef(0.0, 0.0, zoom)
    glRotatef(rot_x, 1, 0, 0)
    glRotatef(rot_y, 0, 1, 0)

    draw_cube()

    glPopMatrix()
    pygame.display.flip()
    pygame.time.wait(10)

    # Show webcam feed
    cv2.imshow("Hand Tracking (Tasks API)", frame)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit
        break

# -----------------------
# Cleanup
# -----------------------
cap.release()
cv2.destroyAllWindows()
pygame.quit()
