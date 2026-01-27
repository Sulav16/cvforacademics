import cv2
import mediapipe as mp
import numpy as np
from math import sin, cos, pi

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# ────────────────────────────────────────────────
# Helper: screen → world at fixed z
# ────────────────────────────────────────────────
def screen_to_world(sx, sy, z, w, h, focal=900, z_offset=800):
    depth = z + z_offset
    factor = focal / depth
    wx = (sx - w / 2) / factor
    wy = (sy - h / 2) / (-factor)
    return np.array([wx, wy, z], dtype=float)


# ────────────────────────────────────────────────
# 3D Shape class with faces for fill 
# ────────────────────────────────────────────────
class Shape3D:
    def __init__(self, shape_type, position, scale=80.0):
        self.shape_type = shape_type
        self.position = np.array(position, dtype=float)
        self.base_scale = scale
        self.scale = scale
        self.angular_velocity = 0.014
        self.rotation_y = 0.0
        self.grab_dist = None
        self.grab_screen_offset = np.zeros(2, dtype=float)

        self.color_pairs = {
            'cube':     ((140,  60, 180), (220, 160, 255)),
            'pyramid':  (( 60, 140,  80), (160, 240, 180)),
            'cylinder': ((180, 100,  40), (255, 200, 140)),
            'cone':     (( 80,  80, 180), (180, 180, 255)),
            'sphere':   ((160, 160,  20), (255, 255, 120)),
        }
        self.color = self.color_pairs.get(shape_type, ((200,200,200),(255,255,255)))

    def update(self, dt=1.0):
        self.rotation_y += self.angular_velocity * dt
        self.rotation_y %= (2 * pi)

    def get_rotation_matrix_y(self):
        c, s = cos(self.rotation_y), sin(self.rotation_y)
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])

    def get_wireframe_and_faces(self):
        verts_local = []
        edges = []
        faces = []

        s = self.scale / 2

        if self.shape_type == 'cube':
            verts_local = [
                [-s,-s,-s],[s,-s,-s],[s,s,-s],[-s,s,-s],
                [-s,-s,s],[s,-s,s],[s,s,s],[-s,s,s]
            ]
            edges = [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)]
            faces = [
                [0,1,2,3],[4,5,6,7],
                [0,1,5,4],[2,3,7,6],
                [1,2,6,5],[0,3,7,4]
            ]

        elif self.shape_type == 'pyramid':
            verts_local = [
                [0, self.scale*0.8, 0],
                [-s,-s,-s],[s,-s,-s],[s,-s,s],[-s,-s,s]
            ]
            edges = [(0,1),(0,2),(0,3),(0,4),(1,2),(2,3),(3,4),(4,1)]
            faces = [[0,1,2],[0,2,3],[0,3,4],[0,4,1],[1,2,3,4]]

        elif self.shape_type == 'cylinder':
            n = 32
            h = self.scale * 0.8
            r = self.scale * 0.4
            for i in range(n):
                ang = i * 2 * pi / n
                x, z = r * cos(ang), r * sin(ang)
                verts_local += [[x, -h/2, z], [x, h/2, z]]
            for i in range(n):
                a = i*2; b = ((i+1)%n)*2
                edges += [(a,b),(a+1,b+1),(a,a+1)]
                faces.append([a,b,b+1,a+1])
            faces.append(list(range(0,2*n,2)))     # bottom
            faces.append(list(range(1,2*n,2)))     # top

        elif self.shape_type == 'cone':
            n = 32
            h = self.scale * 0.95
            r = self.scale * 0.45
            verts_local = [[0, h/2, 0]]
            for i in range(n):
                ang = i * 2 * pi / n
                verts_local.append([r*cos(ang), -h/2, r*sin(ang)])
            for i in range(n):
                ni = (i+1) % n
                edges += [(0, i+1), (i+1, ni+1)]
                faces.append([0, i+1, ni+1])
            faces.append(list(range(1, n+1)))      # base

        elif self.shape_type == 'sphere':
            n = 18
            for i in range(n+1):
                lat = (i / n - 0.5) * pi
                rxy = cos(lat)
                y = sin(lat) * self.scale
                for j in range(n):
                    lon = j * 2 * pi / n
                    x = rxy * cos(lon) * self.scale
                    z = rxy * sin(lon) * self.scale
                    verts_local.append([x, y, z])
            for i in range(n):
                for j in range(n):
                    p1 = i * n + j
                    p2 = i * n + (j + 1) % n
                    edges.append((p1, p2))
                    if i < n:
                        edges.append((p1, (i + 1) * n + j))

        verts_local = np.array(verts_local) if verts_local else np.empty((0,3))
        rot = self.get_rotation_matrix_y()
        verts_world = verts_local @ rot.T + self.position
        return verts_world, edges, faces, self.color


# ────────────────────────────────────────────────
# Projection
# ────────────────────────────────────────────────
def project_points(points, w, h, focal=900, z_offset=800):
    proj = []
    for p in points:
        depth = max(80, p[2] + z_offset)
        factor = focal / depth
        x = p[0] * factor + w / 2
        y = -p[1] * factor + h / 2
        proj.append((int(x), int(y)))
    return proj


# ────────────────────────────────────────────────
# Finger count
# ────────────────────────────────────────────────
def count_extended_fingers(lm, label):
    tips = [4, 8, 12, 16, 20]
    pips = [2, 5, 9, 13, 17]
    count = 0
    tx = lm.landmark[4].x
    if (label == "Right" and tx < lm.landmark[3].x) or \
       (label == "Left"  and tx > lm.landmark[3].x):
        count += 1
    for t, p in zip(tips[1:], pips[1:]):
        if lm.landmark[t].y < lm.landmark[p].y - 0.012:
            count += 1
    return count


# ────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────
cap = cv2.VideoCapture(0)
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

detector = mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.65
)

shapes = []
grabbed = None
grabbed_hand = None

shape_types = ['cube', 'pyramid', 'cylinder', 'cone', 'sphere']
trash_area = (w - 220, h - 180, w, h)

focal = 900
z_offset = 800
default_z = 480.0

grab_threshold    = 95
release_threshold = 135

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = detector.process(rgb)

    pinch_left = pinch_right = None
    left_pinching = right_pinching = False
    right_fingers = 0

    if results.multi_hand_landmarks:
        for hand_lm, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
            mp_drawing.draw_landmarks(frame, hand_lm, mp_hands.HAND_CONNECTIONS)
            label = handedness.classification[0].label

            tx = hand_lm.landmark[4].x * w
            ty = hand_lm.landmark[4].y * h
            ix = hand_lm.landmark[8].x * w
            iy = hand_lm.landmark[8].y * h

            thumb_pt = np.array([tx, ty])
            idx_pt  = np.array([ix, iy])
            dist = np.linalg.norm(thumb_pt - idx_pt)
            pinching = dist < 42
            mid = (thumb_pt + idx_pt) / 2

            if label == "Left":
                pinch_left = mid
                left_pinching = pinching
            else:
                pinch_right = mid
                right_pinching = pinching
                right_fingers = count_extended_fingers(hand_lm, label)

    # ── CREATE ───────────────────────────────────────────────────────
    if len(shapes) == 0 and left_pinching and pinch_left is not None:
        if right_pinching and pinch_right is not None:
            if np.linalg.norm(pinch_left - pinch_right) < 350:
                sx = (pinch_left[0] + pinch_right[0]) / 2
                sy = (pinch_left[1] + pinch_right[1]) / 2
                pos = screen_to_world(sx, sy, default_z, w, h, focal, z_offset)
                shapes = [Shape3D('cube', pos)]

        elif not right_pinching and right_fingers >= 2:
            sx, sy = pinch_left[0], pinch_left[1]
            idx = min(right_fingers - 1, 4)
            pos = screen_to_world(sx, sy, default_z, w, h, focal, z_offset)
            shapes = [Shape3D(shape_types[idx], pos)]

    # ── GRAB / MOVE / ROTATE / SCALE ───────────────────────────────────
    if shapes:
        sh = shapes[0]
        proj_center = project_points([sh.position], w, h, focal, z_offset)[0]
        proj_np = np.array(proj_center)

        can_grab_left  = left_pinching  and pinch_left is not None  and np.linalg.norm(proj_np - pinch_left)  < grab_threshold
        can_grab_right = right_pinching and pinch_right is not None and np.linalg.norm(proj_np - pinch_right) < grab_threshold

        # Release condition (hysteresis)
        if grabbed:
            curr_pinch = pinch_left if grabbed_hand == 'left' else pinch_right
            curr_pinching = left_pinching if grabbed_hand == 'left' else right_pinching
            if not curr_pinching or curr_pinch is None:
                grabbed = None
                grabbed_hand = None
            elif np.linalg.norm(proj_np - curr_pinch) > release_threshold:
                grabbed = None
                grabbed_hand = None

        # Grab logic
        if grabbed is None:
            if can_grab_left:
                grabbed = sh
                grabbed_hand = 'left'
                grabbed.grab_screen_offset = proj_np - pinch_left
                other = pinch_right
                grabbed.grab_dist = np.linalg.norm(pinch_left - other) if other is not None else None

            elif can_grab_right:
                grabbed = sh
                grabbed_hand = 'right'
                grabbed.grab_screen_offset = proj_np - pinch_right
                other = pinch_left
                grabbed.grab_dist = np.linalg.norm(pinch_right - other) if other is not None else None

        # Apply movement / spin / zoom when grabbed
        if grabbed:
            pinch = pinch_left if grabbed_hand == 'left' else pinch_right

            # Very responsive position following
            desired_screen = pinch + grabbed.grab_screen_offset
            target_pos = screen_to_world(desired_screen[0], desired_screen[1],
                                         grabbed.position[2], w, h, focal, z_offset)

            alpha = 0.82          # ← higher = snappier, lower = smoother
            grabbed.position = alpha * target_pos + (1 - alpha) * grabbed.position

            # Spin with smoothing
            spin_hand = pinch_right if grabbed_hand == 'left' else pinch_left
            if spin_hand is not None:
                offset_x = (spin_hand[0] - w/2) / (w/2)
                target_av = offset_x * 0.11
                grabbed.angular_velocity = 0.7 * grabbed.angular_velocity + 0.3 * target_av

            # Two-hand scale (smooth)
            if grabbed.grab_dist is not None:
                other_pinch = pinch_right if grabbed_hand == 'left' else pinch_left
                if other_pinch is not None:
                    curr_dist = np.linalg.norm(pinch - other_pinch)
                    if curr_dist > 25:
                        ratio = curr_dist / grabbed.grab_dist
                        target_scale = sh.base_scale * np.clip(ratio, 0.45, 3.4)
                        sh.scale = 0.84 * sh.scale + 0.16 * target_scale

    else:
        grabbed = None
        grabbed_hand = None

    # ── TRASH (using pinch position — more reliable) ───────────────────
    if grabbed:
        pinch_pt = pinch_left if grabbed_hand == 'left' else pinch_right
        if pinch_pt is not None:
            px, py = pinch_pt
            if trash_area[0] < px < trash_area[2] and trash_area[1] < py < trash_area[3]:
                shapes = []
                grabbed = None
                grabbed_hand = None

    # ── UPDATE & RENDER ────────────────────────────────────────────────
    for sh in shapes:
        sh.update()

    for sh in shapes:
        verts, edges, faces, (dark_col, bright_col) = sh.get_wireframe_and_faces()
        if len(verts) == 0:
            continue
        proj = project_points(verts, w, h, focal, z_offset)

        # Filled faces with shading
        for face in faces:
            pts = np.array([proj[i] for i in face if i < len(proj)], dtype=np.int32)
            if len(pts) < 3:
                continue
            # Simple lighting
            v0, v1, v2 = verts[face[0]], verts[face[1]], verts[face[2]]
            normal = np.cross(v1 - v0, v2 - v0)
            norm_len = np.linalg.norm(normal)
            if norm_len > 1e-6:
                normal /= norm_len
                light = np.array([0, 0, -1])
                intensity = max(0.25, np.dot(normal, light))
                fill_color = tuple(int(c * intensity) for c in bright_col)
                cv2.fillPoly(frame, [pts], fill_color)

        # Edges (dark + bright outline)
        for a, b in edges:
            if 0 <= a < len(proj) and 0 <= b < len(proj):
                cv2.line(frame, proj[a], proj[b], dark_col, 5, cv2.LINE_AA)
        for a, b in edges:
            if 0 <= a < len(proj) and 0 <= b < len(proj):
                cv2.line(frame, proj[a], proj[b], bright_col, 2, cv2.LINE_AA)

    # ── UI ─────────────────────────────────────────────────────────────
    trash_color = (0, 70, 220) if grabbed and pinch_pt is not None and \
                                 trash_area[0] < pinch_pt[0] < trash_area[2] and \
                                 trash_area[1] < pinch_pt[1] < trash_area[3] else (60,60,255)

    cv2.rectangle(frame, (trash_area[0], trash_area[1]), (trash_area[2], trash_area[3]),
                  trash_color, 5 if trash_color != (60,60,255) else 3)
    cv2.putText(frame, "TRASH", (trash_area[0]+45, trash_area[1]+65),
                cv2.FONT_HERSHEY_SIMPLEX, 1.15, trash_color, 2)

    name = shape_types[min(max(right_fingers-1,0),4)] if right_fingers > 0 else "—"
    cv2.putText(frame, f"Right fingers: {right_fingers}  →  {name}",
                (w-300, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,240,100), 2)

    status = "Grabbed" if grabbed else "Free"
    cv2.putText(frame, f"Status: {status} | Double pinch = cube | Left pinch + right fingers = shape",
                (25, h-60), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (210,210,255), 1)
    cv2.putText(frame, "Pinch to grab | Move over red area to delete",
                (25, h-28), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (190,190,240), 1)

    cv2.imshow("Gesture 3D Shapes", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows() 