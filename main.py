import cv2
import mediapipe as mp
import numpy as np
from math import sin, cos, pi

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    min_detection_confidence=0.7,
    min_tracking_confidence=0.65,
    max_num_hands=2
)

# ────────────────────────────────────────────────
# Configuration
# ────────────────────────────────────────────────
MODES = ["Whiteboard", "2D Shapes", "3D Shapes"]
current_mode = 0

colors = [
    (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
    (255, 0, 255), (0, 255, 255), (255, 165, 0)
]
current_color = 0

canvas = None
last_point = None
objects = []           # only one shape

grabbed_shape = None
grabbed_hand = None
grabbed_offset = None
grab_dist = None

trash_size = 220
trash_x = None
trash_y = None

color_box_size = 50
color_gap = 15
color_start_x = 20
color_start_y = 80

mode_btn_width = 160
mode_btn_height = 50
mode_start_x = 20
mode_start_y = 15

focal = 900
z_offset = 800
default_z = 480.0
grab_threshold = 95
release_threshold = 135

MIN_SCALE = 0.35
MAX_SCALE = 4.2

shapes_2d = ['rectangle', 'circle', 'triangle', 'star', 'pentagon']
shapes_3d = ['cube', 'pyramid', 'cylinder', 'cone', 'sphere']

# ────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────
def screen_to_world(sx, sy, z, w, h):
    depth = z + z_offset
    factor = focal / depth
    wx = (sx - w / 2) / factor
    wy = (sy - h / 2) / (-factor)
    return np.array([wx, wy, z], dtype=float)

def project_points(points, w, h):
    proj = []
    for p in points:
        depth = max(80, float(p[2]) + z_offset)
        factor = focal / depth
        x = p[0] * factor + w / 2
        y = -p[1] * factor + h / 2
        proj.append((int(x), int(y)))
    return proj

def point_in_rect(px, py, rx, ry, rw, rh):
    return rx <= px <= rx + rw and ry <= py <= ry + rh

def is_near_trash(px, py):
    if trash_x is None:
        return False
    return trash_x < px < trash_x + trash_size and trash_y < py < trash_y + trash_size

def draw_ui(frame):
    for i, col in enumerate(colors):
        x = color_start_x + i * (color_box_size + color_gap)
        y = color_start_y
        cv2.rectangle(frame, (x, y), (x + color_box_size, y + color_box_size), col, -1)
        if i == current_color:
            cv2.rectangle(frame, (x-5, y-5), (x + color_box_size + 5, y + color_box_size + 5), (255,255,255), 3)

    for i, name in enumerate(MODES):
        x = mode_start_x + i * (mode_btn_width + 20)
        y = mode_start_y
        bg = (100, 220, 100) if i == current_mode else (180, 180, 180)
        cv2.rectangle(frame, (x, y), (x + mode_btn_width, y + mode_btn_height), bg, -1)
        cv2.rectangle(frame, (x, y), (x + mode_btn_width, y + mode_btn_height), (40,40,40), 2)
        tsz = cv2.getTextSize(name, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)[0]
        tx = x + (mode_btn_width - tsz[0]) // 2
        ty = y + (mode_btn_height + tsz[1]) // 2 + 5
        cv2.putText(frame, name, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,0), 2)

def draw_trash(frame, is_grabbing, pinch_pos):
    if trash_x is None:
        return
    col = (0, 70, 220) if is_grabbing and pinch_pos and is_near_trash(pinch_pos[0], pinch_pos[1]) else (60,60,255)
    cv2.rectangle(frame, (trash_x, trash_y), (trash_x + trash_size, trash_y + trash_size), col, -1)
    cv2.rectangle(frame, (trash_x, trash_y), (trash_x + trash_size, trash_y + trash_size), (0,0,220), 5 if col == (0,70,220) else 3)
    cv2.putText(frame, "TRASH", (trash_x + 45, trash_y + 65), cv2.FONT_HERSHEY_SIMPLEX, 1.15, (255,255,255), 2)

def get_projected_center(shape, w, h):
    if isinstance(shape, Shape3D):
        return project_points([shape.position], w, h)[0]
    else:
        return shape['pos']

def draw_2d_shape(frame, obj):
    x, y = map(int, obj['pos'])
    sz = int(140 * obj['scale'])
    half = sz // 2
    c = obj['color']
    typ = obj['type']

    if typ == 'rectangle':
        cv2.rectangle(frame, (x - half, y - half), (x + half, y + half), c, -1)
    elif typ == 'circle':
        cv2.circle(frame, (x, y), half, c, -1)
    elif typ == 'triangle':
        pts = np.array([[x, y - half], [x - half, y + half], [x + half, y + half]], np.int32)
        cv2.fillPoly(frame, [pts], c)
    elif typ == 'star':
        outer = half
        inner = half // 2.5
        pts = []
        for i in range(10):
            r = outer if i % 2 == 0 else inner
            ang = i * pi / 5 - pi / 2
            pts.append([int(x + r * cos(ang)), int(y + r * sin(ang))])
        cv2.fillPoly(frame, [np.array(pts, np.int32)], c)
    elif typ == 'pentagon':
        pts = []
        for i in range(5):
            ang = i * 2 * pi / 5 - pi / 2
            pts.append([int(x + half * cos(ang)), int(y + half * sin(ang))])
        cv2.fillPoly(frame, [np.array(pts, np.int32)], c)

    if obj is grabbed_shape:
        cv2.rectangle(frame, (x - half - 12, y - half - 12), (x + half + 12, y + half + 12), (255,255,255), 4)

# ────────────────────────────────────────────────
# 3D Shape Class – all 5 shapes supported
# ────────────────────────────────────────────────
class Shape3D:
    def __init__(self, shape_type, position, color, scale=120.0):
        self.shape_type = shape_type
        self.position = np.array(position, dtype=float)
        self.scale = scale
        self.angular_velocity = 0.012
        self.rotation_y = 0.0
        self.color = color

    def update(self):
        self.rotation_y += self.angular_velocity
        self.rotation_y %= (2 * pi)

    def get_rotation_matrix_y(self):
        c, s = cos(self.rotation_y), sin(self.rotation_y)
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])

    def get_wireframe_and_faces(self):
        verts_local = []
        edges = []
        faces = []
        s = self.scale / 2
        dark = tuple(int(c * 0.65) for c in self.color)
        bright = self.color

        if self.shape_type == 'cube':
            verts_local = [
                [-s,-s,-s],[s,-s,-s],[s,s,-s],[-s,s,-s],
                [-s,-s,s],[s,-s,s],[s,s,s],[-s,s,s]
            ]
            edges = [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)]
            faces = [[0,1,2,3],[4,5,6,7],[0,1,5,4],[2,3,7,6],[1,2,6,5],[0,3,7,4]]

        elif self.shape_type == 'pyramid':
            verts_local = [
                [0, self.scale*0.8, 0],
                [-s,-s,-s],[s,-s,-s],[s,-s,s],[-s,-s,s]
            ]
            edges = [(0,1),(0,2),(0,3),(0,4),(1,2),(2,3),(3,4),(4,1)]
            faces = [[0,1,2],[0,2,3],[0,3,4],[0,4,1],[1,2,3,4]]

        elif self.shape_type == 'cylinder':
            n = 24
            h = self.scale * 0.9
            r = self.scale * 0.45
            verts_local = []
            for i in range(n):
                ang = i * 2 * pi / n
                x, z = r * cos(ang), r * sin(ang)
                verts_local += [[x, -h/2, z], [x, h/2, z]]
            for i in range(n):
                a = i*2; b = ((i+1)%n)*2
                edges += [(a, b), (a+1, b+1), (a, a+1)]
                faces.append([a, b, b+1, a+1])
            faces.append(list(range(0, 2*n, 2)))   # bottom
            faces.append(list(range(1, 2*n, 2)))   # top

        elif self.shape_type == 'cone':
            n = 24
            h = self.scale * 1.0
            r = self.scale * 0.5
            verts_local = [[0, h/2, 0]]
            for i in range(n):
                ang = i * 2 * pi / n
                verts_local.append([r*cos(ang), -h/2, r*sin(ang)])
            for i in range(n):
                a = i+1; b = ((i+1)%n)+1
                edges += [(0, a), (a, b)]
                faces.append([0, a, b])
            faces.append(list(range(1, n+1)))  # base

        elif self.shape_type == 'sphere':
            n = 12
            verts_local = []
            for i in range(n+1):
                lat = (i / n - 0.5) * pi
                rxy = cos(lat) * self.scale
                y = sin(lat) * self.scale
                for j in range(n):
                    lon = j * 2 * pi / n
                    x = rxy * cos(lon)
                    z = rxy * sin(lon)
                    verts_local.append([x, y, z])
            edges = []
            for i in range(n):
                for j in range(n):
                    a = i * n + j
                    b = i * n + (j+1) % n
                    edges.append((a, b))
                    if i < n-1:
                        edges.append((a, a+n))

        verts_local = np.array(verts_local) if len(verts_local) > 0 else np.empty((0,3))
        rot = self.get_rotation_matrix_y()
        verts_world = verts_local @ rot.T + self.position
        return verts_world, edges, faces, (dark, bright)

# ────────────────────────────────────────────────
# Main loop
# ────────────────────────────────────────────────
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

spawn_cooldown = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w = frame.shape[:2]

    if canvas is None:
        canvas = np.zeros_like(frame)

    if trash_x is None:
        trash_x = w - trash_size - 40
        trash_y = h - trash_size - 40

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    left_pinching = False
    right_pinching = False
    left_pinch_pos = None
    right_pinch_pos = None
    left_fingers = 0
    right_fingers = 0
    active_index_pos = None
    active_fingers = 0

    if results.multi_hand_landmarks:
        for hand_lm, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
            mp_drawing.draw_landmarks(frame, hand_lm, mp_hands.HAND_CONNECTIONS)
            label = handedness.classification[0].label

            tx = hand_lm.landmark[4].x * w
            ty = hand_lm.landmark[4].y * h
            ix = hand_lm.landmark[8].x * w
            iy = hand_lm.landmark[8].y * h

            pinch_dist = np.linalg.norm(np.array([tx, ty]) - np.array([ix, iy]))
            pinching = pinch_dist < 75
            pinch_pos = ((tx + ix)/2, (ty + iy)/2)

            fingers = sum(1 for tip in [8,12,16,20] if hand_lm.landmark[tip].y < hand_lm.landmark[tip-3].y - 0.015)

            if label == "Left":
                left_pinching = pinching
                left_pinch_pos = pinch_pos
                left_fingers = fingers
            else:
                right_pinching = pinching
                right_pinch_pos = pinch_pos
                right_fingers = fingers

            if fingers > active_fingers:
                active_fingers = fingers
                active_index_pos = (int(ix), int(iy))

    # ── UI interaction ──────────────────────────────────────────────
    if active_index_pos:
        sx, sy = active_index_pos
        for i, name in enumerate(MODES):
            x = mode_start_x + i * (mode_btn_width + 20)
            y = mode_start_y
            if point_in_rect(sx, sy, x, y, mode_btn_width, mode_btn_height):
                if current_mode != i:
                    current_mode = i
                    objects = []
                    if current_mode == 0:
                        canvas = np.zeros_like(frame)
                        last_point = None

        for i in range(len(colors)):
            x = color_start_x + i * (color_box_size + color_gap)
            y = color_start_y
            if point_in_rect(sx, sy, x, y, color_box_size, color_box_size):
                current_color = i

    # ── Whiteboard ──────────────────────────────────────────────────
    if current_mode == 0:
        if active_index_pos:
            if 1 <= active_fingers <= 3:
                if last_point:
                    cv2.line(canvas, last_point, active_index_pos, colors[current_color], 9, cv2.LINE_AA)
                last_point = active_index_pos
            else:
                last_point = None

            if active_fingers >= 4:
                cv2.circle(canvas, active_index_pos, 60, (0,0,0), -1)

    # ── Shape modes (only one shape, protected spawn) ───────────────
    else:
        spawn_cooldown = max(0, spawn_cooldown - 1)

        if spawn_cooldown == 0 and not grabbed_shape:
            new_shape = None

            # Double pinch → default shape
            if (left_pinching and right_pinching and
                left_pinch_pos is not None and right_pinch_pos is not None):
                dist = np.linalg.norm(np.array(left_pinch_pos) - np.array(right_pinch_pos))
                if 80 < dist < 520:
                    sx = (left_pinch_pos[0] + right_pinch_pos[0]) / 2
                    sy = (left_pinch_pos[1] + right_pinch_pos[1]) / 2
                    pos = screen_to_world(sx, sy, default_z, w, h) if current_mode == 2 else (sx, sy)

                    if current_mode == 1:
                        new_shape = {'type': shapes_2d[0], 'pos': pos, 'color': colors[current_color], 'scale': 1.0}
                    else:
                        new_shape = Shape3D(shapes_3d[0], pos, colors[current_color])

            # Left pinch + right fingers → unique shapes
            elif (left_pinching and left_pinch_pos is not None and right_fingers >= 2):
                idx = min(right_fingers - 1, 4)
                pos = screen_to_world(left_pinch_pos[0], left_pinch_pos[1], default_z, w, h) if current_mode == 2 else left_pinch_pos

                if current_mode == 1:
                    new_shape = {'type': shapes_2d[idx], 'pos': pos, 'color': colors[current_color], 'scale': 1.0}
                else:
                    new_shape = Shape3D(shapes_3d[idx], pos, colors[current_color])

            if new_shape is not None:
                objects = [new_shape]
                spawn_cooldown = 15

        # ── Grab / Move / Scale / Trash ─────────────────────────────
        if grabbed_shape is not None:
            curr_pinch_pos = left_pinch_pos if grabbed_hand == 'left' else right_pinch_pos
            curr_pinching = left_pinching if grabbed_hand == 'left' else right_pinching

            if not curr_pinching or curr_pinch_pos is None:
                grabbed_shape = grabbed_hand = grabbed_offset = grab_dist = None
            elif np.linalg.norm(np.array(get_projected_center(grabbed_shape, w, h)) - np.array(curr_pinch_pos)) > release_threshold:
                grabbed_shape = grabbed_hand = grabbed_offset = grab_dist = None

        if grabbed_shape is None and (left_pinching or right_pinching):
            candidates = []
            for shape in objects:
                proj = get_projected_center(shape, w, h)
                dist_l = np.linalg.norm(np.array(proj) - np.array(left_pinch_pos)) if left_pinching and left_pinch_pos is not None else float('inf')
                dist_r = np.linalg.norm(np.array(proj) - np.array(right_pinch_pos)) if right_pinching and right_pinch_pos is not None else float('inf')
                min_d = min(dist_l, dist_r)
                if min_d < grab_threshold:
                    hand = 'left' if dist_l <= dist_r else 'right'
                    candidates.append((min_d, shape, hand, proj))

            if candidates:
                candidates.sort(key=lambda x: x[0])
                _, grabbed_shape, grabbed_hand, proj_center = candidates[0]
                pinch_pos = left_pinch_pos if grabbed_hand == 'left' else right_pinch_pos
                grabbed_offset = np.array(proj_center) - np.array(pinch_pos)
                other = right_pinch_pos if grabbed_hand == 'left' else left_pinch_pos
                grab_dist = np.linalg.norm(np.array(pinch_pos) - np.array(other)) if other is not None else None

        if grabbed_shape is not None:
            pinch_pos = left_pinch_pos if grabbed_hand == 'left' else right_pinch_pos
            if pinch_pos is not None:
                desired = np.array(pinch_pos) + grabbed_offset

                if current_mode == 2:
                    target = screen_to_world(desired[0], desired[1], grabbed_shape.position[2], w, h)
                    alpha = 0.93
                    grabbed_shape.position = alpha * target + (1 - alpha) * grabbed_shape.position
                else:
                    grabbed_shape['pos'] = tuple(desired)

                other = right_pinch_pos if grabbed_hand == 'left' else left_pinch_pos
                if other is not None and grab_dist is not None:
                    curr_dist = np.linalg.norm(np.array(pinch_pos) - np.array(other))
                    if curr_dist > 50:
                        raw_ratio = curr_dist / grab_dist
                        damped_ratio = 0.6 * raw_ratio + 0.4 * 1.0
                        damped_ratio = np.clip(damped_ratio, 0.75, 1.35)

                        if current_mode == 2:
                            new_scale = grabbed_shape.scale * damped_ratio
                            grabbed_shape.scale = np.clip(new_scale, MIN_SCALE * 80, MAX_SCALE * 80)
                        else:
                            new_scale = grabbed_shape['scale'] * damped_ratio
                            grabbed_shape['scale'] = np.clip(new_scale, MIN_SCALE, MAX_SCALE)

                if is_near_trash(pinch_pos[0], pinch_pos[1]):
                    objects = []
                    grabbed_shape = grabbed_hand = grabbed_offset = grab_dist = None

    # ── Rendering ───────────────────────────────────────────────────
    draw_ui(frame)
    active_pinch = left_pinch_pos if left_pinching else right_pinch_pos
    draw_trash(frame, grabbed_shape is not None, active_pinch)

    if current_mode in (1, 2) and objects:
        obj = objects[0]

        if current_mode == 2:
            verts, edges, faces, (dark_col, bright_col) = obj.get_wireframe_and_faces()
            if len(verts) > 0:
                proj = project_points(verts, w, h)
                for face in faces:
                    pts = [proj[i] for i in face if i < len(proj)]
                    if len(pts) < 3: continue
                    pts = np.array(pts, np.int32)
                    v0, v1, v2 = verts[face[0]], verts[face[1]], verts[face[2]]
                    normal = np.cross(v1 - v0, v2 - v0)
                    norm_len = np.linalg.norm(normal)
                    if norm_len > 1e-6:
                        normal /= norm_len
                        intensity = max(0.3, np.dot(normal, [0,0,-1]))
                        fill_color = tuple(int(c * intensity) for c in bright_col)
                        cv2.fillPoly(frame, [pts], fill_color)
                for a, b in edges:
                    if 0 <= a < len(proj) and 0 <= b < len(proj):
                        cv2.line(frame, proj[a], proj[b], dark_col, 5, cv2.LINE_AA)
                        cv2.line(frame, proj[a], proj[b], bright_col, 2, cv2.LINE_AA)
                obj.update()

        else:
            draw_2d_shape(frame, obj)

    display = cv2.addWeighted(frame, 0.65, canvas, 0.35, 0) if current_mode == 0 else frame.copy()
    cv2.putText(display, f"Mode: {MODES[current_mode]}", (30, h-40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (220,220,255), 2)

    cv2.imshow("Gesture Canvas", display)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()