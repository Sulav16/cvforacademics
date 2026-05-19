"""
Gesture Canvas — v2 Final
──────────────────────────
New in v2:
  • Say shape name to spawn it  ("cube", "sphere", "circle", "star" etc)
  • Multiple shapes on screen at once — each independently grabbable
  • No auto-erase on mode switch — canvas and shapes persist
  • Smooth zoom — interpolated, no jumps
  • Say "delete" to remove grabbed shape
  • Say color name while holding shape → recolors that shape
  • Shape counter HUD
  • Drag any individual shape to trash to delete just that one

Install:
  pip install opencv-python mediapipe numpy SpeechRecognition pyaudio easyocr torch torchvision
"""

import cv2
import mediapipe as mp
import numpy as np
from math import sin, cos, pi
import threading
import queue

# ── Optional imports ──────────────────────────────────────────────────────────
try:
    import speech_recognition as sr  # type: ignore
    VOICE_AVAILABLE = True
except ImportError:
    VOICE_AVAILABLE = False
    print("[WARN] SpeechRecognition not installed — voice commands disabled.")

try:
    import easyocr  # type: ignore
    ocr_reader = easyocr.Reader(['en'], gpu=False, verbose=False)
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False
    print("[WARN] EasyOCR not installed — OCR feature disabled.")

# ────────────────────────────────────────────────
# Configuration
# ────────────────────────────────────────────────
MODES = ["Whiteboard", "2D Shapes", "3D Shapes"]

COLORS = [
    (255, 0,   0  ),   # 0 Blue
    (0,   255, 0  ),   # 1 Green
    (0,   0,   255),   # 2 Red
    (255, 255, 0  ),   # 3 Cyan
    (255, 0,   255),   # 4 Magenta
    (0,   255, 255),   # 5 Yellow
    (0,   165, 255),   # 6 Orange
    (255, 255, 255),   # 7 White
]
COLOR_NAMES = ['Blue', 'Green', 'Red', 'Cyan', 'Magenta', 'Yellow', 'Orange', 'White']

DRAW_THICKNESS    = 9
ERASE_RADIUS      = 60
CANVAS_BLEND      = 0.35
PINCH_THRESHOLD   = 75
GRAB_THRESHOLD    = 95
RELEASE_THRESHOLD = 135
FOCAL             = 900
Z_OFFSET          = 800
DEFAULT_Z         = 480.0
MIN_SCALE_2D      = 0.25
MAX_SCALE_2D      = 5.0
MIN_SCALE_3D      = MIN_SCALE_2D * 80
MAX_SCALE_3D      = MAX_SCALE_2D * 80
ZOOM_SMOOTH       = 0.12     # interpolation factor for smooth zoom (lower = smoother)

COLOR_BOX_SIZE    = 45
COLOR_GAP         = 12
COLOR_START_X     = 20
COLOR_START_Y     = 80
MODE_BTN_WIDTH    = 155
MODE_BTN_HEIGHT   = 48
MODE_START_X      = 20
MODE_START_Y      = 15
TRASH_SIZE        = 180
SPAWN_COOLDOWN    = 18

SHAPES_2D = ['rectangle', 'circle', 'triangle', 'star', 'pentagon']
SHAPES_3D = ['cube', 'pyramid', 'cylinder', 'cone', 'sphere']

# ────────────────────────────────────────────────
# State
# ────────────────────────────────────────────────
current_mode   = 0
current_color  = 0
canvas         = None
last_point     = None
objects        = []          # list of shape dicts or Shape3D — now supports multiple

grabbed_shape  = None
grabbed_hand   = None
grabbed_offset = None
grab_dist      = None        # fixed reference distance at grab time
grab_scale_ref = None        # fixed scale at grab time
target_scale   = None        # desired scale — actual scale lerps toward this

spawn_cooldown = 0
trash_x = trash_y = None

ocr_overlays   = []
ocr_running    = False

voice_queue    = queue.Queue()
voice_status   = ""
voice_status_ttl = 0

# ────────────────────────────────────────────────
# MediaPipe
# ────────────────────────────────────────────────
mp_hands   = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    min_detection_confidence=0.7,
    min_tracking_confidence=0.65,
    max_num_hands=2,
)

# ────────────────────────────────────────────────
# Voice commands
# ────────────────────────────────────────────────
VOICE_COMMANDS = {
    # Colors
    "blue":        ("color",  0),
    "green":       ("color",  1),
    "red":         ("color",  2),
    "cyan":        ("color",  3),
    "magenta":     ("color",  4),
    "yellow":      ("color",  5),
    "orange":      ("color",  6),
    "white":       ("color",  7),
    # Modes
    "whiteboard":  ("mode",   0),
    "two d":       ("mode",   1),
    "2d":          ("mode",   1),
    "three d":     ("mode",   2),
    "3d":          ("mode",   2),
    # Actions
    "clear text":   ("action", "clear_text"),
    "clear drawing":("action", "clear_text"),
    "erase text":   ("action", "clear_text"),
    "clear shapes": ("action", "clear_shapes"),
    "remove shapes":("action", "clear_shapes"),
    "clear all":    ("action", "clear_all"),
    "erase all":    ("action", "clear_all"),
    "clear":        ("action", "clear_text"),   # default clear = text only
    "erase":        ("action", "clear_text"),
    "read":        ("action", "ocr"),
    "ocr":         ("action", "ocr"),
    "snap":        ("action", "snap"),
    "delete":      ("action", "delete"),
    "remove":      ("action", "delete"),
    # 2D shapes
    "rectangle":   ("spawn",  ("2d", "rectangle")),
    "square":      ("spawn",  ("2d", "rectangle")),
    "circle":      ("spawn",  ("2d", "circle")),
    "triangle":    ("spawn",  ("2d", "triangle")),
    "star":        ("spawn",  ("2d", "star")),
    "pentagon":    ("spawn",  ("2d", "pentagon")),
    # 3D shapes
    "cube":        ("spawn",  ("3d", "cube")),
    "pyramid":     ("spawn",  ("3d", "pyramid")),
    "cylinder":    ("spawn",  ("3d", "cylinder")),
    "cone":        ("spawn",  ("3d", "cone")),
    "sphere":      ("spawn",  ("3d", "sphere")),
}

def voice_listener():
    if not VOICE_AVAILABLE:
        return
    recognizer = sr.Recognizer()
    recognizer.energy_threshold        = 300    # fixed sensitivity — more reliable
    recognizer.dynamic_energy_threshold = False  # disable auto-adjustment
    recognizer.pause_threshold          = 0.5    # faster response after speaking
    mic = sr.Microphone()
    with mic as source:
        recognizer.adjust_for_ambient_noise(source, duration=0.3)  # shorter calibration
    while True:
        try:
            with mic as source:
                audio = recognizer.listen(source, timeout=5, phrase_time_limit=3)
            text = recognizer.recognize_google(audio).lower()
            print(f"[VOICE] Heard: '{text}'")   # shows in terminal so you know what was picked up
            for keyword, cmd in VOICE_COMMANDS.items():
                if keyword in text:
                    voice_queue.put(cmd)
                    print(f"[VOICE] Command matched: '{keyword}'")
                    break
        except sr.WaitTimeoutError:
            pass   # no speech detected — normal, keep looping
        except sr.UnknownValueError:
            pass   # couldn't understand — normal
        except Exception as e:
            print(f"[VOICE] Error: {e}")

if VOICE_AVAILABLE:
    threading.Thread(target=voice_listener, daemon=True).start()

# ────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────
def reset_grab():
    global grabbed_shape, grabbed_hand, grabbed_offset
    global grab_dist, grab_scale_ref, target_scale
    grabbed_shape = grabbed_hand = grabbed_offset = None
    grab_dist = grab_scale_ref = target_scale = None

def switch_mode(new_mode):
    """Switch mode — canvas and shapes NEVER cleared on switch."""
    global current_mode
    reset_grab()
    current_mode = new_mode

def spawn_shape(kind, shape_type, cx, cy, w, h):
    """Spawn a shape at screen position cx, cy."""
    if kind == "2d":
        return {
            'type':  shape_type,
            'pos':   [float(cx), float(cy)],
            'color': COLORS[current_color],
            'scale': 1.0,
        }
    else:
        pos = screen_to_world(cx, cy, DEFAULT_Z, w, h)
        return Shape3D(shape_type, pos, COLORS[current_color])

def screen_to_world(sx, sy, z, w, h):
    depth  = z + Z_OFFSET
    factor = FOCAL / depth
    return np.array([(sx-w/2)/factor, (sy-h/2)/(-factor), z], dtype=float)

def project_points(points, w, h):
    out = []
    for p in points:
        depth  = max(80, float(p[2]) + Z_OFFSET)
        factor = FOCAL / depth
        out.append((int(p[0]*factor + w/2), int(-p[1]*factor + h/2)))
    return out

def point_in_rect(px, py, rx, ry, rw, rh):
    return rx <= px <= rx+rw and ry <= py <= ry+rh

def is_near_trash(px, py):
    return (trash_x is not None and
            trash_x < px < trash_x+TRASH_SIZE and
            trash_y < py < trash_y+TRASH_SIZE)

def get_projected_center(shape, w, h):
    if isinstance(shape, Shape3D):
        return project_points([shape.position], w, h)[0]
    return (int(shape['pos'][0]), int(shape['pos'][1]))

def get_shape_scale(shape):
    return shape.scale if isinstance(shape, Shape3D) else shape['scale']

def set_shape_scale(shape, val):
    if isinstance(shape, Shape3D):
        shape.scale = float(np.clip(val, MIN_SCALE_3D, MAX_SCALE_3D))
    else:
        shape['scale'] = float(np.clip(val, MIN_SCALE_2D, MAX_SCALE_2D))

# ────────────────────────────────────────────────
# Auto shape snapping
# ────────────────────────────────────────────────
def snap_canvas_strokes(cnv, color):
    gray    = cv2.cvtColor(cnv, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (9,9), 2)
    circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, dp=1.2,
                               minDist=50, param1=80, param2=30,
                               minRadius=20, maxRadius=400)
    if circles is not None:
        snapped = np.zeros_like(cnv)
        cx, cy, r = map(int, circles[0][0])
        cv2.circle(snapped, (cx, cy), r, color, DRAW_THICKNESS, cv2.LINE_AA)
        return snapped

    _, thresh   = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return cnv

    largest = max(contours, key=cv2.contourArea)
    peri    = cv2.arcLength(largest, True)
    approx  = cv2.approxPolyDP(largest, 0.04*peri, True)
    n       = len(approx)
    snapped = np.zeros_like(cnv)

    if n == 3:
        cv2.polylines(snapped, [approx], True, color, DRAW_THICKNESS, cv2.LINE_AA)
    elif n == 4:
        x, y, bw, bh = cv2.boundingRect(approx)
        aspect = bw/bh if bh else 1
        if 0.85 < aspect < 1.15:
            s = (bw+bh)//2; cx2, cy2 = x+bw//2, y+bh//2
            pts = np.array([[cx2-s//2,cy2-s//2],[cx2+s//2,cy2-s//2],
                             [cx2+s//2,cy2+s//2],[cx2-s//2,cy2+s//2]])
            cv2.polylines(snapped, [pts], True, color, DRAW_THICKNESS, cv2.LINE_AA)
        else:
            cv2.rectangle(snapped, (x,y), (x+bw,y+bh), color, DRAW_THICKNESS, cv2.LINE_AA)
    elif n >= 5:
        cv2.polylines(snapped, [cv2.convexHull(approx)], True, color, DRAW_THICKNESS, cv2.LINE_AA)
    else:
        return cnv
    return snapped

# ────────────────────────────────────────────────
# EasyOCR
# ────────────────────────────────────────────────
def run_ocr_async(cnv):
    global ocr_running
    if not OCR_AVAILABLE or ocr_running:
        return
    def _task():
        global ocr_running
        try:
            gray    = cv2.cvtColor(cnv, cv2.COLOR_BGR2GRAY)
            results = ocr_reader.readtext(gray, detail=1, paragraph=False)
            for bbox, text, conf in results:
                if conf < 0.3 or not text.strip():
                    continue
                xs = [p[0] for p in bbox]; ys = [p[1] for p in bbox]
                ocr_overlays.append([text.strip(),
                                     (int(sum(xs)/4), int(sum(ys)/4)), 120])
        finally:
            ocr_running = False
    ocr_running = True
    threading.Thread(target=_task, daemon=True).start()

# ────────────────────────────────────────────────
# UI
# ────────────────────────────────────────────────
def draw_ui(frame):
    for i, col in enumerate(COLORS):
        x = COLOR_START_X + i*(COLOR_BOX_SIZE+COLOR_GAP)
        y = COLOR_START_Y
        cv2.rectangle(frame, (x,y), (x+COLOR_BOX_SIZE, y+COLOR_BOX_SIZE), col, -1)
        if i == current_color:
            cv2.rectangle(frame, (x-4,y-4),
                          (x+COLOR_BOX_SIZE+4, y+COLOR_BOX_SIZE+4), (255,255,255), 3)

    for i, name in enumerate(MODES):
        x  = MODE_START_X + i*(MODE_BTN_WIDTH+18)
        y  = MODE_START_Y
        bg = (80, 200, 80) if i == current_mode else (170,170,170)
        cv2.rectangle(frame, (x,y), (x+MODE_BTN_WIDTH, y+MODE_BTN_HEIGHT), bg, -1)
        cv2.rectangle(frame, (x,y), (x+MODE_BTN_WIDTH, y+MODE_BTN_HEIGHT), (40,40,40), 2)
        tsz = cv2.getTextSize(name, cv2.FONT_HERSHEY_SIMPLEX, 0.85, 2)[0]
        cv2.putText(frame, name,
                    (x+(MODE_BTN_WIDTH-tsz[0])//2, y+(MODE_BTN_HEIGHT+tsz[1])//2+4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0,0,0), 2)

def draw_trash(frame, is_grabbing, pinch_pos):
    if trash_x is None: return
    hovering = is_grabbing and pinch_pos and is_near_trash(pinch_pos[0], pinch_pos[1])
    col = (0,60,200) if hovering else (50,50,240)
    cv2.rectangle(frame, (trash_x,trash_y),
                  (trash_x+TRASH_SIZE, trash_y+TRASH_SIZE), col, -1)
    cv2.rectangle(frame, (trash_x,trash_y),
                  (trash_x+TRASH_SIZE, trash_y+TRASH_SIZE), (0,0,180), 4)
    cv2.putText(frame, "TRASH",
                (trash_x+28, trash_y+55),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255), 2)

def draw_hud(frame, h, w):
    global voice_status_ttl
    cv2.rectangle(frame, (0,h-42), (w,h), (25,25,25), -1)
    hints = "S=Snap  O=OCR  C=Clear canvas  X=Clear shapes  Z=Clear all  ESC=Quit"
    if VOICE_AVAILABLE:
        hints += "  | Voice: 'clear text' / 'clear shapes' / 'clear all'"
    cv2.putText(frame, hints, (12, h-12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.46, (150,150,255), 1)

    # Shape counter
    if objects:
        cv2.putText(frame, f"Shapes: {len(objects)}",
                    (w-160, h-55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200,255,200), 2)

    # Voice status
    if voice_status and voice_status_ttl > 0:
        cv2.putText(frame, f"Voice: {voice_status}",
                    (12, h-52),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.72, (0,255,180), 2)
        voice_status_ttl -= 1

def draw_ocr_overlays(frame):
    for overlay in ocr_overlays:
        text, (cx,cy), ttl = overlay
        alpha = min(1.0, ttl/30.0)
        cv2.putText(frame, text, (cx,cy),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.1,
                    (0, int(255*alpha), int(120*alpha)), 3, cv2.LINE_AA)
        overlay[2] -= 1
    ocr_overlays[:] = [o for o in ocr_overlays if o[2] > 0]

# ────────────────────────────────────────────────
# 2D shape drawing
# ────────────────────────────────────────────────
def draw_2d_shape(frame, obj):
    x, y = int(obj['pos'][0]), int(obj['pos'][1])
    sz   = int(140 * obj['scale'])
    half = max(4, sz//2)
    c    = obj['color']
    typ  = obj['type']

    if typ == 'rectangle':
        cv2.rectangle(frame, (x-half,y-half), (x+half,y+half), c, -1)
    elif typ == 'circle':
        cv2.circle(frame, (x,y), half, c, -1)
    elif typ == 'triangle':
        pts = np.array([[x,y-half],[x-half,y+half],[x+half,y+half]], np.int32)
        cv2.fillPoly(frame, [pts], c)
    elif typ == 'star':
        outer, inner = half, max(2, half//2)
        pts = [[int(x+(outer if i%2==0 else inner)*cos(i*pi/5-pi/2)),
                int(y+(outer if i%2==0 else inner)*sin(i*pi/5-pi/2))]
               for i in range(10)]
        cv2.fillPoly(frame, [np.array(pts, np.int32)], c)
    elif typ == 'pentagon':
        pts = [[int(x+half*cos(i*2*pi/5-pi/2)),
                int(y+half*sin(i*2*pi/5-pi/2))]
               for i in range(5)]
        cv2.fillPoly(frame, [np.array(pts, np.int32)], c)

    if obj is grabbed_shape:
        cv2.rectangle(frame, (x-half-10,y-half-10),
                      (x+half+10,y+half+10), (255,255,255), 3)

# ────────────────────────────────────────────────
# 3D Shape class
# ────────────────────────────────────────────────
class Shape3D:
    def __init__(self, shape_type, position, color, scale=120.0):
        self.shape_type       = shape_type
        self.position         = np.array(position, dtype=float)
        self.scale            = float(scale)
        self._last_scale      = float(scale)
        self.angular_velocity = 0.012
        self.rotation_y       = 0.0
        self.color            = color
        self._cache           = None

    def update(self):
        self.rotation_y = (self.rotation_y + self.angular_velocity) % (2*pi)

    def _build(self):
        s = self.scale/2
        v, e, f = [], [], []

        if self.shape_type == 'cube':
            v = [[-s,-s,-s],[s,-s,-s],[s,s,-s],[-s,s,-s],
                 [-s,-s,s],[s,-s,s],[s,s,s],[-s,s,s]]
            e = [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)]
            f = [[0,1,2,3],[4,5,6,7],[0,1,5,4],[2,3,7,6],[1,2,6,5],[0,3,7,4]]

        elif self.shape_type == 'pyramid':
            v = [[0,self.scale*0.8,0],[-s,-s,-s],[s,-s,-s],[s,-s,s],[-s,-s,s]]
            e = [(0,1),(0,2),(0,3),(0,4),(1,2),(2,3),(3,4),(4,1)]
            f = [[0,1,2],[0,2,3],[0,3,4],[0,4,1],[1,2,3,4]]

        elif self.shape_type == 'cylinder':
            n=24; h2=self.scale*0.9; r=self.scale*0.45
            for i in range(n):
                ang=i*2*pi/n; x2,z=r*cos(ang),r*sin(ang)
                v+=[[x2,-h2/2,z],[x2,h2/2,z]]
            for i in range(n):
                a=i*2; b=((i+1)%n)*2
                e+=[(a,b),(a+1,b+1),(a,a+1)]; f.append([a,b,b+1,a+1])
            f.append(list(range(0,2*n,2))); f.append(list(range(1,2*n,2)))

        elif self.shape_type == 'cone':
            n=24; h2=self.scale; r=self.scale*0.5
            v=[[0,h2/2,0]]
            for i in range(n):
                ang=i*2*pi/n; v.append([r*cos(ang),-h2/2,r*sin(ang)])
            for i in range(n):
                a=i+1; b=((i+1)%n)+1
                e+=[(0,a),(a,b)]; f.append([0,a,b])
            f.append(list(range(1,n+1)))

        elif self.shape_type == 'sphere':
            n=12
            for i in range(n+1):
                lat=(i/n-0.5)*pi; rxy=cos(lat)*self.scale; y2=sin(lat)*self.scale
                for j in range(n):
                    lon=j*2*pi/n; v.append([rxy*cos(lon),y2,rxy*sin(lon)])
            for i in range(n):
                for j in range(n):
                    a=i*n+j; b=i*n+(j+1)%n
                    e.append((a,b))
                    if i<n-1: e.append((a,a+n))

        self._cache     = (np.array(v,dtype=float), e, f)
        self._last_scale = self.scale

    def get_wireframe_and_faces(self):
        if self._cache is None or self._last_scale != self.scale:
            self._build()
        vl, edges, faces = self._cache
        c,s   = cos(self.rotation_y), sin(self.rotation_y)
        rot   = np.array([[c,0,s],[0,1,0],[-s,0,c]])
        vw    = vl @ rot.T + self.position
        dark  = tuple(int(ch*0.65) for ch in self.color)
        return vw, edges, faces, (dark, self.color)

# ────────────────────────────────────────────────
# Camera init
# ────────────────────────────────────────────────
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
ret, _f      = cap.read()
cap_h, cap_w = _f.shape[:2] if ret else (720,1280)

canvas  = np.zeros((cap_h, cap_w, 3), dtype=np.uint8)
trash_x = cap_w - TRASH_SIZE - 30
trash_y = cap_h - TRASH_SIZE - 30

# ────────────────────────────────────────────────
# Main loop
# ────────────────────────────────────────────────
while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break

    frame = cv2.flip(frame, 1)
    h, w  = frame.shape[:2]

    # ── Hand detection ───────────────────────────────────────────────
    rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    left_pinching = right_pinching = False
    left_pinch_pos = right_pinch_pos = None
    left_fingers = right_fingers = 0
    active_index_pos = None
    active_fingers   = 0

    if results.multi_hand_landmarks:
        for hand_lm, handedness in zip(results.multi_hand_landmarks,
                                        results.multi_handedness):
            mp_drawing.draw_landmarks(frame, hand_lm, mp_hands.HAND_CONNECTIONS)
            label = handedness.classification[0].label
            tx = hand_lm.landmark[4].x*w;  ty = hand_lm.landmark[4].y*h
            ix = hand_lm.landmark[8].x*w;  iy = hand_lm.landmark[8].y*h
            pinching  = np.linalg.norm([tx-ix, ty-iy]) < PINCH_THRESHOLD
            pinch_pos = ((tx+ix)/2, (ty+iy)/2)
            fingers   = sum(1 for tip in [8,12,16,20]
                            if hand_lm.landmark[tip].y < hand_lm.landmark[tip-3].y-0.015)

            if label == "Left":
                left_pinching=pinching; left_pinch_pos=pinch_pos; left_fingers=fingers
            else:
                right_pinching=pinching; right_pinch_pos=pinch_pos; right_fingers=fingers

            if label=="Right" or active_index_pos is None:
                if fingers >= active_fingers:
                    active_fingers=fingers; active_index_pos=(int(ix),int(iy))

    # ── Voice commands ───────────────────────────────────────────────
    while not voice_queue.empty():
        kind, val = voice_queue.get_nowait()

        if kind == "color":
            current_color = val
            # If holding a shape, recolor it immediately
            if grabbed_shape is not None:
                if isinstance(grabbed_shape, Shape3D):
                    grabbed_shape.color = COLORS[val]
                else:
                    grabbed_shape['color'] = COLORS[val]
            voice_status = f"Color → {COLOR_NAMES[val]}"

        elif kind == "mode":
            switch_mode(val)
            voice_status = f"Mode → {MODES[val]}"

        elif kind == "spawn":
            dim, shape_type = val
            cx, cy = w//2, h//2
            if dim == "2d":
                objects.append(spawn_shape("2d", shape_type, cx, cy, w, h))
            else:
                objects.append(spawn_shape("3d", shape_type, cx, cy, w, h))
            voice_status = f"Spawned {shape_type}"

        elif kind == "action":
            if val == "clear_text":
                canvas     = np.zeros((h, w, 3), dtype=np.uint8)
                last_point = None
                voice_status = "Canvas cleared"

            elif val == "clear_shapes":
                objects.clear()
                reset_grab()
                voice_status = "Shapes cleared"

            elif val == "clear_all":
                canvas     = np.zeros((h, w, 3), dtype=np.uint8)
                last_point = None
                objects.clear()
                reset_grab()
                voice_status = "Everything cleared"

            elif val == "delete":
                if grabbed_shape is not None:
                    if grabbed_shape in objects:
                        objects.remove(grabbed_shape)
                    reset_grab()
                    voice_status = "Shape deleted"
                elif objects:
                    objects.pop()
                    voice_status = "Shape removed"

            elif val == "ocr":
                if current_mode == 0:
                    run_ocr_async(canvas.copy())
                    voice_status = "Reading text..."

            elif val == "snap":
                if current_mode == 0:
                    canvas = snap_canvas_strokes(canvas, COLORS[current_color])
                    last_point = None
                    voice_status = "Shape snapped!"

        voice_status_ttl = 90

    # ── UI interaction ───────────────────────────────────────────────
    if active_index_pos:
        sx, sy = active_index_pos
        for i in range(len(MODES)):
            x = MODE_START_X + i*(MODE_BTN_WIDTH+18)
            if point_in_rect(sx,sy, x,MODE_START_Y, MODE_BTN_WIDTH,MODE_BTN_HEIGHT):
                if current_mode != i:
                    switch_mode(i)
        for i in range(len(COLORS)):
            x = COLOR_START_X + i*(COLOR_BOX_SIZE+COLOR_GAP)
            if point_in_rect(sx,sy, x,COLOR_START_Y, COLOR_BOX_SIZE,COLOR_BOX_SIZE):
                current_color = i

    # ── Whiteboard ───────────────────────────────────────────────────
    if current_mode == 0:
        if active_index_pos:
            if 1 <= active_fingers <= 3:
                if last_point:
                    cv2.line(canvas, last_point, active_index_pos,
                             COLORS[current_color], DRAW_THICKNESS, cv2.LINE_AA)
                last_point = active_index_pos
            else:
                last_point = None
            if active_fingers == 4:
                cv2.circle(canvas, active_index_pos, ERASE_RADIUS, (0,0,0), -1)
            if active_fingers >= 5:
                canvas = snap_canvas_strokes(canvas, COLORS[current_color])
                last_point = None

    # ── Shape interaction (works in ALL modes) ───────────────────────
    spawn_cooldown = max(0, spawn_cooldown-1)

    # Spawn via gesture (only when NOT in whiteboard drawing mode)
    if current_mode != 0 and spawn_cooldown == 0 and not grabbed_shape:
        new_shape = None

        if left_pinching and right_pinching and left_pinch_pos and right_pinch_pos:
            dist = np.linalg.norm(np.array(left_pinch_pos)-np.array(right_pinch_pos))
            if 80 < dist < 520:
                cx2 = (left_pinch_pos[0]+right_pinch_pos[0])/2
                cy2 = (left_pinch_pos[1]+right_pinch_pos[1])/2
                if current_mode == 1:
                    new_shape = spawn_shape("2d", SHAPES_2D[0], cx2, cy2, w, h)
                else:
                    new_shape = spawn_shape("3d", SHAPES_3D[0], cx2, cy2, w, h)

        elif left_pinching and left_pinch_pos and right_fingers >= 2:
            idx = min(right_fingers-1, 4)
            if current_mode == 1:
                new_shape = spawn_shape("2d", SHAPES_2D[idx],
                                        left_pinch_pos[0], left_pinch_pos[1], w, h)
            else:
                new_shape = spawn_shape("3d", SHAPES_3D[idx],
                                        left_pinch_pos[0], left_pinch_pos[1], w, h)

        if new_shape is not None:
            objects.append(new_shape)
            spawn_cooldown = SPAWN_COOLDOWN

    # Release grabbed shape
    if grabbed_shape is not None:
        curr_pp       = left_pinch_pos  if grabbed_hand=='left' else right_pinch_pos
        curr_pinching = left_pinching   if grabbed_hand=='left' else right_pinching
        if not curr_pinching or curr_pp is None:
            reset_grab()
        else:
            center  = get_projected_center(grabbed_shape, w, h)
            too_far = np.linalg.norm(np.array(center)-np.array(curr_pp)) > RELEASE_THRESHOLD
            if too_far:
                reset_grab()

    # New grab — pick closest shape (works in all modes)
    if grabbed_shape is None and objects and (left_pinching or right_pinching):
        candidates = []
        for shape in objects:
            proj  = get_projected_center(shape, w, h)
            dl    = (np.linalg.norm(np.array(proj)-np.array(left_pinch_pos))
                     if left_pinching and left_pinch_pos else float('inf'))
            dr    = (np.linalg.norm(np.array(proj)-np.array(right_pinch_pos))
                     if right_pinching and right_pinch_pos else float('inf'))
            mind  = min(dl, dr)
            if mind < GRAB_THRESHOLD:
                candidates.append((mind, shape,
                                   'left' if dl<=dr else 'right', proj))
        if candidates:
            candidates.sort(key=lambda x: x[0])
            _, grabbed_shape, grabbed_hand, proj_c = candidates[0]
            pp_g           = left_pinch_pos if grabbed_hand=='left' else right_pinch_pos
            grabbed_offset = np.array(proj_c) - np.array(pp_g)
            other_pp       = right_pinch_pos if grabbed_hand=='left' else left_pinch_pos
            grab_dist      = (np.linalg.norm(np.array(pp_g)-np.array(other_pp))
                              if other_pp else None)
            grab_scale_ref = get_shape_scale(grabbed_shape)
            target_scale   = grab_scale_ref

    # Move + smooth zoom (works in all modes)
    if grabbed_shape is not None:
        pp_g = left_pinch_pos if grabbed_hand=='left' else right_pinch_pos
        if pp_g:
            desired = np.array(pp_g) + grabbed_offset

            # Move
            if isinstance(grabbed_shape, Shape3D):
                target_pos = screen_to_world(desired[0], desired[1],
                                              grabbed_shape.position[2], w, h)
                grabbed_shape.position = 0.92*target_pos + 0.08*grabbed_shape.position
            else:
                grabbed_shape['pos'] = desired.tolist()

            # Smooth zoom
            other_pp = right_pinch_pos if grabbed_hand=='left' else left_pinch_pos
            if (other_pp is not None and
                    grab_dist is not None and
                    grab_dist > 50 and
                    grab_scale_ref is not None):
                curr_dist     = np.linalg.norm(np.array(pp_g)-np.array(other_pp))
                raw_ratio     = np.clip(curr_dist / grab_dist, 0.15, 5.0)
                desired_scale = grab_scale_ref * raw_ratio
                if isinstance(grabbed_shape, Shape3D):
                    desired_scale = np.clip(desired_scale, MIN_SCALE_3D, MAX_SCALE_3D)
                else:
                    desired_scale = np.clip(desired_scale, MIN_SCALE_2D, MAX_SCALE_2D)
                current_s = get_shape_scale(grabbed_shape)
                set_shape_scale(grabbed_shape,
                                current_s + ZOOM_SMOOTH*(desired_scale - current_s))

            # Trash — delete only this shape
            if is_near_trash(pp_g[0], pp_g[1]):
                if grabbed_shape in objects:
                    objects.remove(grabbed_shape)
                reset_grab()

    # ── Render ───────────────────────────────────────────────────────
    # Step 1: always composite canvas onto frame first (preserves whiteboard across modes)
    display = cv2.addWeighted(frame, 1-CANVAS_BLEND, canvas, CANVAS_BLEND, 0)

    # Step 2: always draw ALL shapes on top regardless of mode
    # 3D shapes
    for obj in objects:
        if not isinstance(obj, Shape3D):
            continue
        verts, edges, faces, (dark_col, bright_col) = obj.get_wireframe_and_faces()
        if len(verts) == 0:
            continue
        proj = project_points(verts, w, h)
        for face in faces:
            pts = [proj[i] for i in face if i < len(proj)]
            if len(pts) < 3:
                continue
            pts_arr = np.array(pts, np.int32)
            v0,v1,v2 = verts[face[0]],verts[face[1]],verts[face[2]]
            normal = np.cross(v1-v0, v2-v0)
            nlen   = np.linalg.norm(normal)
            if nlen > 1e-6:
                normal   /= nlen
                intensity = max(0.3, np.dot(normal,[0,0,-1]))
                cv2.fillPoly(display, [pts_arr],
                             tuple(int(c*intensity) for c in bright_col))
        for a,b in edges:
            if 0<=a<len(proj) and 0<=b<len(proj):
                cv2.line(display, proj[a], proj[b], dark_col,   5, cv2.LINE_AA)
                cv2.line(display, proj[a], proj[b], bright_col, 2, cv2.LINE_AA)
        obj.update()

    # 2D shapes
    for obj in objects:
        if isinstance(obj, Shape3D):
            continue
        draw_2d_shape(display, obj)

    # Step 3: OCR overlays
    draw_ocr_overlays(display)

    # Step 4: UI on top of everything
    draw_ui(display)
    active_pinch = left_pinch_pos if left_pinching else right_pinch_pos
    draw_trash(display, grabbed_shape is not None, active_pinch)
    draw_hud(display, h, w)
    cv2.putText(display, f"Mode: {MODES[current_mode]}",
                (30, h-50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (220,220,255), 2)

    cv2.imshow("Gesture Canvas", display)

    key = cv2.waitKey(1) & 0xFF
    if key == 27:
        break
    elif key == ord('s') and current_mode == 0:
        canvas = snap_canvas_strokes(canvas, COLORS[current_color])
        last_point = None
    elif key == ord('o') and current_mode == 0:
        run_ocr_async(canvas.copy())
    elif key == ord('c'):                  # C = clear canvas only
        canvas = np.zeros((h,w,3), dtype=np.uint8); last_point = None
    elif key == ord('x'):                  # X = clear shapes only
        objects.clear(); reset_grab()
    elif key == ord('z'):                  # Z = clear everything
        canvas = np.zeros((h,w,3), dtype=np.uint8); last_point = None
        objects.clear(); reset_grab()

cap.release()
cv2.destroyAllWindows()