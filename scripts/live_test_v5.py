"""
Live webcam behaviour detection — V5 (Generalised + Phone-Aware).

Uses V4 ML model + rule-based phone-detection cascade.
Works from any angle, distance, or environment.

Labels:
  • STUDYING     — focused, hands on desk / near study objects
  • DISTRACTED   — looking away, leaning, head turned
  • USING PHONE  — phone detected + head/hands/gaze aligned
  • NO PERSON    — nobody in frame

Keys: q=Quit  m=Toggle XGB/RF  y=Toggle YOLO  d=Toggle debug
"""

import os
import sys
import time
import cv2
import mediapipe as mp
import numpy as np
import joblib
import warnings
from collections import deque
from ultralytics import YOLO

warnings.filterwarnings("ignore")

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")

xgb_path = os.path.join(MODELS_DIR, "xgb_behaviour_model_v4.pkl")
rf_path = os.path.join(MODELS_DIR, "rf_behaviour_model_v4.pkl")
le_path = os.path.join(MODELS_DIR, "label_encoder_v4.pkl")

# ─── Load models ──────────────────────────────────────────────────────────────
print("Loading V4 generalised models...")
xgb_model = joblib.load(xgb_path)
rf_model = joblib.load(rf_path)
le = joblib.load(le_path)
print(f"  Classes: {list(le.classes_)}")

use_xgb = True
show_yolo = True
show_debug = False

# ─── YOLO ─────────────────────────────────────────────────────────────────────
print("Loading YOLOv8n...")
yolo_model = YOLO("yolov8n.pt")

YOLO_CLASSES = {
    "person": 0, "phone": 67, "book": 73, "laptop": 63,
    "remote": 65, "mouse": 64, "keyboard": 66, "tv": 62,
    "bottle": 39, "cup": 41, "scissors": 76, "clock": 74,
    "backpack": 24, "handbag": 26,
}
YOLO_ID_TO_NAME = {v: k for k, v in YOLO_CLASSES.items()}
DISTRACTOR_IDS = {67, 65, 62, 74}
STUDY_IDS = {73, 63, 64, 66, 76}

YOLO_BOX_COLORS = {
    "distractor": (0, 0, 255), "study": (0, 200, 0),
    "person": (255, 200, 0), "other": (200, 200, 0),
}

# ─── MediaPipe ────────────────────────────────────────────────────────────────
mp_face_mesh = mp.solutions.face_mesh
mp_hands = mp.solutions.hands
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False, max_num_faces=1, refine_landmarks=True,
    min_detection_confidence=0.3, min_tracking_confidence=0.4,
)
hands_det = mp_hands.Hands(
    static_image_mode=False, max_num_hands=2,
    min_detection_confidence=0.3, min_tracking_confidence=0.4,
)
pose_det = mp_pose.Pose(
    static_image_mode=False, model_complexity=2,
    min_detection_confidence=0.3, min_tracking_confidence=0.4,
)

# ─── Landmark groups ──────────────────────────────────────────────────────────
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
LEFT_EYEBROW = [70, 63, 105, 66, 107]
RIGHT_EYEBROW = [300, 293, 334, 296, 336]
FACE_OVAL = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323,
             361, 288, 397, 365, 379, 378, 400, 377, 152, 148,
             176, 149, 150, 136, 172, 58, 132, 93, 234, 127,
             162, 21, 54, 103, 67, 109]

MODEL_POINTS_3D = np.array([
    (0.0, 0.0, 0.0), (0.0, -330.0, -65.0),
    (-225.0, 170.0, -135.0), (225.0, 170.0, -135.0),
    (-150.0, -150.0, -125.0), (150.0, -150.0, -125.0),
], dtype=np.float64)
FACE_2D_INDICES = [1, 152, 33, 263, 61, 291]

# ─── V4 feature columns (must match training order) ──────────────────────────
FEATURE_COLS = [
    # F. Image-level (8)
    "img_edge_density", "img_edge_center", "img_texture_var",
    "img_skin_ratio", "img_skin_upper",
    "img_gradient_mean", "img_phone_rect_count",
    # E. YOLO (14) — note: phone_detected and phone_count are in phone-specific
    "yolo_person_detected", "yolo_person_conf", "yolo_person_area",
    "yolo_phone_conf", "yolo_phone_area",
    "yolo_book_detected", "yolo_book_area",
    "yolo_laptop_detected", "yolo_laptop_area",
    "yolo_num_objects", "yolo_num_distractors", "yolo_num_study_items",
    "yolo_bottle_or_cup", "yolo_any_distractor",
    # A. Face geometry (16)  — face_detected is separate
    "face_detected",
    "ear_left", "ear_right", "ear_avg", "mar",
    "head_pitch_deg", "head_yaw_deg", "head_roll_deg",
    "head_pitch_geom", "head_yaw_geom",
    "gaze_x", "gaze_y",
    "face_aspect_ratio", "face_area_rel",
    "eyebrow_left_dist_norm", "eyebrow_right_dist_norm",
    # B. Hand-face (14)
    "num_hands", "hand_near_face", "hand_near_ear",
    "hands_in_phone_zone", "hand_below_face",
    "hand_face_dist_min", "hand_face_dist_avg",
    "hand_face_angle_avg", "hand_face_angle_min",
    "hand_spread_avg", "hand_spread_max",
    "hand_wrist_y_avg", "hand_wrist_y_max", "hand_wrist_y_min",
    # C. Pose (16)
    "pose_detected", "shoulder_roll_deg", "shoulder_width_norm",
    "head_shoulder_dist_norm", "body_lean_angle",
    "left_elbow_angle", "right_elbow_angle",
    "left_wrist_above_shoulder", "right_wrist_above_shoulder",
    "arm_symmetry", "left_wrist_shoulder_dist_norm", "right_wrist_shoulder_dist_norm",
    "nose_shoulder_offset_norm", "torso_visible",
    "left_hand_raised_high", "right_hand_raised_high",
    # D. Phone-specific (10)
    "phone_detected", "phone_count", "phone_box_area",
    "phone_face_dist_norm", "phone_face_angle_deg", "phone_below_face",
    "gaze_phone_align", "head_towards_phone", "wrist_near_phone",
    "phone_holding_score",
]


# ═══════════════════════════════════════════════════════════════════════════════
#  HELPER FUNCTIONS (mirrors extract_features_v4.py)
# ═══════════════════════════════════════════════════════════════════════════════

def _pt(lm, idx):
    return np.array([lm[idx].x, lm[idx].y, lm[idx].z])

def _pt2(lm, idx):
    return np.array([lm[idx].x, lm[idx].y])

def _pose_pt(landmarks, idx):
    i = idx.value if hasattr(idx, "value") else idx
    lm = landmarks[i]
    return np.array([lm.x, lm.y, lm.z])

def _angle_between(v1, v2):
    cos = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
    return np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))

def _safe_div(a, b, default=0.0):
    return a / b if abs(b) > 1e-8 else default


# --- Face ---
def compute_ear(lm, indices):
    p = [_pt2(lm, i) for i in indices]
    vert = np.linalg.norm(p[1] - p[5]) + np.linalg.norm(p[2] - p[4])
    horiz = np.linalg.norm(p[0] - p[3])
    return _safe_div(vert, 2.0 * horiz)

def compute_mar(lm):
    top, bot = _pt2(lm, 13), _pt2(lm, 14)
    left, right = _pt2(lm, 61), _pt2(lm, 291)
    return _safe_div(np.linalg.norm(top - bot), np.linalg.norm(left - right))

def compute_head_pose_solvepnp(lm, h, w):
    img_pts = np.array([[lm[i].x * w, lm[i].y * h] for i in FACE_2D_INDICES], dtype=np.float64)
    focal = w
    cam_matrix = np.array([[focal, 0, w/2], [0, focal, h/2], [0, 0, 1]], dtype=np.float64)
    ok, rvec, tvec = cv2.solvePnP(MODEL_POINTS_3D, img_pts, cam_matrix, np.zeros((4,1)), flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        return 0.0, 0.0, 0.0
    rmat, _ = cv2.Rodrigues(rvec)
    angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)
    return float(angles[0]), float(angles[1]), float(angles[2])

def compute_head_pose_geom(lm):
    nose, chin = _pt(lm, 1), _pt(lm, 152)
    le, re = _pt(lm, 33), _pt(lm, 263)
    eye_mid = (le + re) / 2.0
    ed = np.linalg.norm(le[:2] - re[:2])
    if ed < 1e-6: return 0.0, 0.0, 0.0
    yaw = (nose[0] - eye_mid[0]) / ed
    fh = np.linalg.norm(eye_mid[:2] - chin[:2])
    pitch = _safe_div(nose[1] - eye_mid[1], fh)
    roll = np.degrees(np.arctan2(re[1] - le[1], re[0] - le[0]))
    return pitch, yaw, roll

def compute_gaze(lm):
    try:
        li, ri = _pt2(lm, 468), _pt2(lm, 473)
        lc = (_pt2(lm, 133) + _pt2(lm, 33)) / 2.0
        rc = (_pt2(lm, 362) + _pt2(lm, 263)) / 2.0
        lw = np.linalg.norm(_pt2(lm, 133) - _pt2(lm, 33))
        rw = np.linalg.norm(_pt2(lm, 362) - _pt2(lm, 263))
        if lw < 1e-6 or rw < 1e-6: return 0.0, 0.0
        g = ((li - lc) / lw + (ri - rc) / rw) / 2.0
        return float(g[0]), float(g[1])
    except Exception:
        return 0.0, 0.0


# ═══════════════════════════════════════════════════════════════════════════════
#  YOLO
# ═══════════════════════════════════════════════════════════════════════════════

def run_yolo(image):
    """Run YOLO → features dict, phone_boxes, raw detections list."""
    h, w = image.shape[:2]
    img_area = h * w

    feat = {
        "yolo_person_detected": 0, "yolo_person_conf": 0.0, "yolo_person_area": 0.0,
        "yolo_phone_conf": 0.0, "yolo_phone_area": 0.0,
        "yolo_book_detected": 0, "yolo_book_area": 0.0,
        "yolo_laptop_detected": 0, "yolo_laptop_area": 0.0,
        "yolo_num_objects": 0, "yolo_num_distractors": 0, "yolo_num_study_items": 0,
        "yolo_bottle_or_cup": 0, "yolo_any_distractor": 0,
    }
    phone_boxes = []
    detections = []
    person_detected = False

    results = yolo_model(image, verbose=False, conf=0.25)
    if len(results) == 0 or results[0].boxes is None or len(results[0].boxes) == 0:
        return feat, phone_boxes, detections, person_detected

    boxes = results[0].boxes
    cls_ids = boxes.cls.cpu().numpy().astype(int)
    confs = boxes.conf.cpu().numpy()
    xyxy = boxes.xyxy.cpu().numpy()

    feat["yolo_num_objects"] = int(len(cls_ids))

    for i, cid in enumerate(cls_ids):
        x1, y1, x2, y2 = xyxy[i]
        box_area = (x2 - x1) * (y2 - y1) / img_area
        conf = float(confs[i])
        name = YOLO_ID_TO_NAME.get(cid, None)
        if name is None:
            continue
        detections.append((int(x1), int(y1), int(x2), int(y2), cid, conf, name))

        if cid == YOLO_CLASSES["person"]:
            person_detected = True
            if conf > feat["yolo_person_conf"]:
                feat["yolo_person_detected"] = 1
                feat["yolo_person_conf"] = conf
                feat["yolo_person_area"] = box_area
        elif cid == YOLO_CLASSES["phone"]:
            if conf > feat["yolo_phone_conf"]:
                feat["yolo_phone_conf"] = conf
                feat["yolo_phone_area"] = box_area
            phone_boxes.append((x1/w, y1/h, x2/w, y2/h, conf))
        elif cid == YOLO_CLASSES["book"]:
            if conf > feat["yolo_book_area"]:
                feat["yolo_book_detected"] = 1
                feat["yolo_book_area"] = box_area
        elif cid == YOLO_CLASSES["laptop"]:
            if conf > feat["yolo_laptop_area"]:
                feat["yolo_laptop_detected"] = 1
                feat["yolo_laptop_area"] = box_area
        if cid in (YOLO_CLASSES["bottle"], YOLO_CLASSES["cup"]):
            feat["yolo_bottle_or_cup"] = 1
        if cid in DISTRACTOR_IDS:
            feat["yolo_num_distractors"] += 1
            feat["yolo_any_distractor"] = 1
        if cid in STUDY_IDS:
            feat["yolo_num_study_items"] += 1

    return feat, phone_boxes, detections, person_detected


# ═══════════════════════════════════════════════════════════════════════════════
#  LIVE FEATURE EXTRACTION (matches V4 training features exactly)
# ═══════════════════════════════════════════════════════════════════════════════

def extract_live_features(bgr_frame, rgb_frame):
    """Extract all V4 features for one frame.
    Returns (face_res, hand_res, pose_res, feature_vec, yolo_dets, phone_boxes, yolo_person)."""
    h, w = bgr_frame.shape[:2]
    feat = {k: 0.0 for k in FEATURE_COLS}

    # --- Image-level (8) ---
    gray = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2HSV)
    edges = cv2.Canny(gray, 50, 150)
    feat["img_edge_density"] = edges.mean() / 255.0
    cw1, cw2 = w // 4, 3 * w // 4
    ch1, ch2 = h // 4, 3 * h // 4
    feat["img_edge_center"] = edges[ch1:ch2, cw1:cw2].mean() / 255.0
    feat["img_texture_var"] = cv2.Laplacian(gray, cv2.CV_64F).var() / 10000.0
    lower_skin = np.array([0, 20, 70])
    upper_skin = np.array([20, 255, 255])
    skin_mask = cv2.inRange(hsv, lower_skin, upper_skin)
    feat["img_skin_ratio"] = skin_mask.mean() / 255.0
    feat["img_skin_upper"] = skin_mask[:h//2, :].mean() / 255.0
    gx_s = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy_s = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    feat["img_gradient_mean"] = np.sqrt(gx_s**2 + gy_s**2).mean() / 255.0
    _, bright = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)
    h3 = h // 3
    bright_upper = bright[:2*h3, :]
    contours, _ = cv2.findContours(bright_upper, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    rect_count = 0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 500:
            _, _, cw_r, ch_r = cv2.boundingRect(cnt)
            aspect = cw_r / ch_r if ch_r > 0 else 0
            if 0.3 <= aspect <= 0.85 or 1.2 <= aspect <= 2.5:
                rect_count += 1
    feat["img_phone_rect_count"] = rect_count

    # --- YOLO (14) ---
    yolo_feat, phone_boxes, yolo_dets, yolo_person = run_yolo(bgr_frame)
    feat.update(yolo_feat)

    # --- MediaPipe ---
    face_results = face_mesh.process(rgb_frame)
    hand_results = hands_det.process(rgb_frame)
    pose_results = pose_det.process(rgb_frame)

    face_detected = (face_results.multi_face_landmarks is not None and
                     len(face_results.multi_face_landmarks) > 0)
    feat["face_detected"] = int(face_detected)

    face_lm = face_results.multi_face_landmarks[0].landmark if face_detected else None
    face_w_val, face_h_val = 0.0, 0.0

    # --- A. Face geometry (16) ---
    if face_detected:
        feat["ear_left"] = compute_ear(face_lm, LEFT_EYE)
        feat["ear_right"] = compute_ear(face_lm, RIGHT_EYE)
        feat["ear_avg"] = (feat["ear_left"] + feat["ear_right"]) / 2.0
        feat["mar"] = compute_mar(face_lm)

        p_s, y_s, r_s = compute_head_pose_solvepnp(face_lm, h, w)
        feat["head_pitch_deg"] = p_s
        feat["head_yaw_deg"] = y_s
        feat["head_roll_deg"] = r_s

        p_g, y_g, _ = compute_head_pose_geom(face_lm)
        feat["head_pitch_geom"] = p_g
        feat["head_yaw_geom"] = y_g

        gx, gy = compute_gaze(face_lm)
        feat["gaze_x"] = gx
        feat["gaze_y"] = gy

        xs = [face_lm[i].x for i in FACE_OVAL]
        ys = [face_lm[i].y for i in FACE_OVAL]
        face_w_val = max(xs) - min(xs)
        face_h_val = max(ys) - min(ys)
        feat["face_aspect_ratio"] = _safe_div(face_w_val, face_h_val)
        feat["face_area_rel"] = face_w_val * face_h_val

        lb_y = np.mean([face_lm[i].y for i in LEFT_EYEBROW])
        le_y = np.mean([face_lm[i].y for i in LEFT_EYE])
        rb_y = np.mean([face_lm[i].y for i in RIGHT_EYEBROW])
        re_y = np.mean([face_lm[i].y for i in RIGHT_EYE])
        feat["eyebrow_left_dist_norm"] = _safe_div(le_y - lb_y, face_h_val)
        feat["eyebrow_right_dist_norm"] = _safe_div(re_y - rb_y, face_h_val)

    # --- B. Hand-face (14) ---
    num_hands = 0
    hand_near_face = 0
    hand_near_ear = 0
    hands_in_phone_zone = 0
    hand_below_face_count = 0
    hand_face_angles = []
    hand_face_dists = []
    hand_spread_list = []
    hand_wrist_ys = []

    if hand_results.multi_hand_landmarks:
        num_hands = len(hand_results.multi_hand_landmarks)

    if face_detected and hand_results.multi_hand_landmarks:
        face_cx = (min(face_lm[i].x for i in FACE_OVAL) + max(face_lm[i].x for i in FACE_OVAL)) / 2.0
        face_cy = (min(face_lm[i].y for i in FACE_OVAL) + max(face_lm[i].y for i in FACE_OVAL)) / 2.0
        face_diag = np.sqrt(face_w_val**2 + face_h_val**2) + 1e-8

        for hand_lm in hand_results.multi_hand_landmarks:
            hx = np.mean([l.x for l in hand_lm.landmark])
            hy = np.mean([l.y for l in hand_lm.landmark])
            wrist = np.array([hand_lm.landmark[0].x, hand_lm.landmark[0].y])
            hand_wrist_ys.append(wrist[1])
            dist = np.sqrt((hx - face_cx)**2 + (hy - face_cy)**2)
            dist_norm = dist / face_diag
            hand_face_dists.append(dist_norm)
            if dist_norm < 1.2:
                hand_near_face += 1
            ear_l = _pt2(face_lm, 234)
            ear_r = _pt2(face_lm, 454)
            d_el = np.linalg.norm(np.array([hx, hy]) - ear_l) / face_diag
            d_er = np.linalg.norm(np.array([hx, hy]) - ear_r) / face_diag
            if min(d_el, d_er) < 0.5:
                hand_near_ear += 1
            angle = np.degrees(np.arctan2(hy - face_cy, hx - face_cx))
            hand_face_angles.append(angle)
            chin_y = face_lm[152].y
            if hy > chin_y and hy < chin_y + face_h_val * 2.5:
                hands_in_phone_zone += 1
            if hy > face_cy:
                hand_below_face_count += 1
            pts = np.array([[l.x, l.y] for l in hand_lm.landmark])
            hand_spread_list.append(np.std(pts, axis=0).mean())
    elif not face_detected and hand_results.multi_hand_landmarks:
        for hand_lm in hand_results.multi_hand_landmarks:
            wrist = np.array([hand_lm.landmark[0].x, hand_lm.landmark[0].y])
            hand_wrist_ys.append(wrist[1])
            pts = np.array([[l.x, l.y] for l in hand_lm.landmark])
            hand_spread_list.append(np.std(pts, axis=0).mean())

    feat["num_hands"] = num_hands
    feat["hand_near_face"] = hand_near_face
    feat["hand_near_ear"] = hand_near_ear
    feat["hands_in_phone_zone"] = hands_in_phone_zone
    feat["hand_below_face"] = hand_below_face_count
    feat["hand_face_dist_min"] = min(hand_face_dists) if hand_face_dists else 0.0
    feat["hand_face_dist_avg"] = np.mean(hand_face_dists) if hand_face_dists else 0.0
    feat["hand_face_angle_avg"] = np.mean(hand_face_angles) if hand_face_angles else 0.0
    feat["hand_face_angle_min"] = min(hand_face_angles) if hand_face_angles else 0.0
    feat["hand_spread_avg"] = np.mean(hand_spread_list) if hand_spread_list else 0.0
    feat["hand_spread_max"] = max(hand_spread_list) if hand_spread_list else 0.0
    feat["hand_wrist_y_avg"] = np.mean(hand_wrist_ys) if hand_wrist_ys else 0.0
    feat["hand_wrist_y_max"] = max(hand_wrist_ys) if hand_wrist_ys else 0.0
    feat["hand_wrist_y_min"] = min(hand_wrist_ys) if hand_wrist_ys else 0.0

    # --- C. Pose (16) ---
    pose_lm = pose_results.pose_landmarks
    if pose_lm is not None:
        plm = pose_lm.landmark
        ls = _pose_pt(plm, mp_pose.PoseLandmark.LEFT_SHOULDER)
        rs = _pose_pt(plm, mp_pose.PoseLandmark.RIGHT_SHOULDER)
        nose_p = _pose_pt(plm, mp_pose.PoseLandmark.NOSE)
        le_p = _pose_pt(plm, mp_pose.PoseLandmark.LEFT_ELBOW)
        re_p = _pose_pt(plm, mp_pose.PoseLandmark.RIGHT_ELBOW)
        lw_p = _pose_pt(plm, mp_pose.PoseLandmark.LEFT_WRIST)
        rw_p = _pose_pt(plm, mp_pose.PoseLandmark.RIGHT_WRIST)
        lh_p = _pose_pt(plm, mp_pose.PoseLandmark.LEFT_HIP)
        rh_p = _pose_pt(plm, mp_pose.PoseLandmark.RIGHT_HIP)

        shoulder_mid = (ls + rs) / 2.0
        sw = np.linalg.norm(ls[:2] - rs[:2])
        hip_mid = (lh_p + rh_p) / 2.0
        torso_h = np.linalg.norm(shoulder_mid[:2] - hip_mid[:2])

        feat["pose_detected"] = 1.0
        feat["shoulder_roll_deg"] = np.degrees(np.arctan2(rs[1] - ls[1], rs[0] - ls[0]))
        feat["shoulder_width_norm"] = _safe_div(sw, torso_h)
        feat["head_shoulder_dist_norm"] = _safe_div(np.linalg.norm(nose_p[:2] - shoulder_mid[:2]), sw)

        torso_vec = hip_mid[:2] - shoulder_mid[:2]
        vertical = np.array([0.0, 1.0])
        feat["body_lean_angle"] = _angle_between(torso_vec, vertical) if np.linalg.norm(torso_vec) > 1e-6 else 0.0

        def elbow_angle(shoulder, elbow, wrist):
            v1 = shoulder[:2] - elbow[:2]
            v2 = wrist[:2] - elbow[:2]
            return _angle_between(v1, v2) if (np.linalg.norm(v1) > 1e-6 and np.linalg.norm(v2) > 1e-6) else 180.0

        feat["left_elbow_angle"] = elbow_angle(ls, le_p, lw_p)
        feat["right_elbow_angle"] = elbow_angle(rs, re_p, rw_p)
        feat["left_wrist_above_shoulder"] = float(lw_p[1] < ls[1])
        feat["right_wrist_above_shoulder"] = float(rw_p[1] < rs[1])
        feat["arm_symmetry"] = _safe_div(abs(lw_p[1] - rw_p[1]), sw)
        feat["left_wrist_shoulder_dist_norm"] = _safe_div(np.linalg.norm(lw_p[:2] - ls[:2]), sw)
        feat["right_wrist_shoulder_dist_norm"] = _safe_div(np.linalg.norm(rw_p[:2] - rs[:2]), sw)
        feat["nose_shoulder_offset_norm"] = _safe_div(abs(nose_p[0] - shoulder_mid[0]), sw)
        lh_vis = plm[mp_pose.PoseLandmark.LEFT_HIP.value].visibility
        rh_vis = plm[mp_pose.PoseLandmark.RIGHT_HIP.value].visibility
        feat["torso_visible"] = float(lh_vis > 0.5 and rh_vis > 0.5)
        feat["left_hand_raised_high"] = float(lw_p[1] < nose_p[1])
        feat["right_hand_raised_high"] = float(rw_p[1] < nose_p[1])

    # --- D. Phone-specific (10) ---
    feat["phone_detected"] = int(len(phone_boxes) > 0)
    feat["phone_count"] = len(phone_boxes)
    best_phone = None
    best_area = 0.0
    for box in phone_boxes:
        bx1, by1, bx2, by2, conf = box
        area = (bx2 - bx1) * (by2 - by1)
        if area > best_area:
            best_area = area
            best_phone = box
    feat["phone_box_area"] = best_area

    if best_phone is not None and face_detected:
        bx1, by1, bx2, by2, _ = best_phone
        pcx = (bx1 + bx2) / 2.0
        pcy = (by1 + by2) / 2.0
        face_cx = (min(face_lm[i].x for i in FACE_OVAL) + max(face_lm[i].x for i in FACE_OVAL)) / 2.0
        face_cy = (min(face_lm[i].y for i in FACE_OVAL) + max(face_lm[i].y for i in FACE_OVAL)) / 2.0
        face_diag = np.sqrt(face_w_val**2 + face_h_val**2) + 1e-8
        nose = _pt2(face_lm, 1)

        feat["phone_face_dist_norm"] = np.sqrt((face_cx - pcx)**2 + (face_cy - pcy)**2) / face_diag
        feat["phone_face_angle_deg"] = np.degrees(np.arctan2(pcy - face_cy, pcx - face_cx))
        feat["phone_below_face"] = float(pcy > face_cy)

        gx, gy = compute_gaze(face_lm)
        gaze_vec = np.array([gx, gy])
        phone_dir = np.array([pcx - nose[0], pcy - nose[1]])
        pdl = np.linalg.norm(phone_dir)
        if pdl > 1e-6 and np.linalg.norm(gaze_vec) > 1e-6:
            feat["gaze_phone_align"] = float(np.dot(gaze_vec, phone_dir / pdl))

        le, re = _pt(face_lm, 33), _pt(face_lm, 263)
        eye_mid = (le + re) / 2.0
        ed = np.linalg.norm(le[:2] - re[:2])
        if ed > 1e-6:
            head_dir_x = (nose[0] - eye_mid[0]) / ed
            chin = _pt2(face_lm, 152)
            face_h_local = np.linalg.norm(eye_mid[:2] - chin)
            head_dir_y = _safe_div(nose[1] - eye_mid[1], face_h_local)
            phone_dx = pcx - nose[0]
            phone_dy = pcy - nose[1]
            yaw_ok = (phone_dx * head_dir_x > 0) or abs(phone_dx) < 0.10
            pitch_ok = (phone_dy > 0 and head_dir_y > 0) or abs(phone_dy) < 0.08
            feat["head_towards_phone"] = float(yaw_ok and pitch_ok)

        if pose_lm is not None:
            plm = pose_lm.landmark
            lw_pos = _pose_pt(plm, mp_pose.PoseLandmark.LEFT_WRIST)[:2]
            rw_pos = _pose_pt(plm, mp_pose.PoseLandmark.RIGHT_WRIST)[:2]
            phone_pt = np.array([pcx, pcy])
            sw_val = np.linalg.norm(_pose_pt(plm, mp_pose.PoseLandmark.LEFT_SHOULDER)[:2] -
                                    _pose_pt(plm, mp_pose.PoseLandmark.RIGHT_SHOULDER)[:2])
            margin = sw_val * 0.3 if sw_val > 1e-6 else 0.06
            d_l = np.linalg.norm(lw_pos - phone_pt)
            d_r = np.linalg.norm(rw_pos - phone_pt)
            feat["wrist_near_phone"] = float(min(d_l, d_r) < margin)

    signals = (
        feat["phone_detected"] * 2.0 +
        feat["head_towards_phone"] * 1.5 +
        feat["gaze_phone_align"] * 1.0 +
        feat["wrist_near_phone"] * 1.5 +
        feat["phone_below_face"] * 0.5
    )
    feat["phone_holding_score"] = min(signals / 5.0, 1.0)

    feature_vec = np.array([feat[c] for c in FEATURE_COLS])
    return (face_results, hand_results, pose_results,
            feature_vec, yolo_dets, phone_boxes, yolo_person or (feat["yolo_person_detected"] == 1))


# ═══════════════════════════════════════════════════════════════════════════════
#  PHONE RULE-BASED OVERRIDE (backup when YOLO misses phone)
# ═══════════════════════════════════════════════════════════════════════════════

def rule_phone_without_yolo(face_lm, pose_lm, h, w):
    """
    Detect phone usage even when YOLO doesn't detect a phone:
      - Head pitched down (looking at something in hands)
      - Hands in phone-zone (between chin and chest)
      - Elbow bent ~40-90° (holding posture)
      - Gaze directed downward
    Returns (is_phone, reason_str).
    """
    if face_lm is None or pose_lm is None:
        return False, ""

    plm = pose_lm.landmark

    # Head pitch (solvePnP)
    pitch_deg, yaw_deg, _ = compute_head_pose_solvepnp(face_lm, h, w)
    looking_down = pitch_deg < -10  # looking down > 10°

    # Gaze direction
    gx, gy = compute_gaze(face_lm)
    gaze_down = gy > 0.02  # gaze shifted downward

    # Elbow angle (bent arm)
    ls = _pose_pt(plm, mp_pose.PoseLandmark.LEFT_SHOULDER)
    rs = _pose_pt(plm, mp_pose.PoseLandmark.RIGHT_SHOULDER)
    le = _pose_pt(plm, mp_pose.PoseLandmark.LEFT_ELBOW)
    re = _pose_pt(plm, mp_pose.PoseLandmark.RIGHT_ELBOW)
    lw = _pose_pt(plm, mp_pose.PoseLandmark.LEFT_WRIST)
    rw = _pose_pt(plm, mp_pose.PoseLandmark.RIGHT_WRIST)

    def elbow_ang(s, e, wr):
        v1 = s[:2] - e[:2]
        v2 = wr[:2] - e[:2]
        if np.linalg.norm(v1) < 1e-6 or np.linalg.norm(v2) < 1e-6:
            return 180.0
        return _angle_between(v1, v2)

    la = elbow_ang(ls, le, lw)
    ra = elbow_ang(rs, re, rw)
    arm_bent = la < 100 or ra < 100  # typical phone hold: 40-90°

    # Wrist in phone zone (above waist, below chin, near centre)
    chin_y = face_lm[152].y
    nose_y = face_lm[1].y
    lw_y, rw_y = lw[1], rw[1]
    wrist_in_zone = ((chin_y < lw_y < chin_y + 0.3) or
                     (chin_y < rw_y < chin_y + 0.3))

    reasons = []
    if looking_down:
        reasons.append(f"Head down {pitch_deg:.0f}°")
    if gaze_down:
        reasons.append("Gaze down")
    if arm_bent:
        reasons.append(f"Arm bent ({min(la,ra):.0f}°)")
    if wrist_in_zone:
        reasons.append("Wrist in phone zone")

    # Need at least 3 of 4 signals
    score = sum([looking_down, gaze_down, arm_bent, wrist_in_zone])
    if score >= 3:
        return True, " + ".join(reasons)
    return False, ""


# ═══════════════════════════════════════════════════════════════════════════════
#  DRAW HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def draw_yolo_boxes(frame, detections):
    for (x1, y1, x2, y2, cid, conf, name) in detections:
        if cid == YOLO_CLASSES["person"]:
            color = YOLO_BOX_COLORS["person"]
        elif cid in DISTRACTOR_IDS:
            color = YOLO_BOX_COLORS["distractor"]
        elif cid in STUDY_IDS:
            color = YOLO_BOX_COLORS["study"]
        else:
            color = YOLO_BOX_COLORS["other"]
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        label_text = f"{name} {conf:.0%}"
        (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
        cv2.putText(frame, label_text, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)


LABEL_COLORS = {
    "no person":   (180, 180, 180),
    "studying":    (0, 200, 0),
    "distracted":  (0, 100, 255),
    "using phone": (0, 0, 255),
    "phone using": (0, 0, 255),
}


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN LOOP
# ═══════════════════════════════════════════════════════════════════════════════

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("ERROR: Cannot open webcam")
    sys.exit(1)

buffer = deque(maxlen=7)
phone_hold_frames = 0       # consecutive frames with phone signal
PHONE_HOLD_THRESHOLD = 4    # need 4+ frames to confirm phone
current_label = "Detecting..."
confidence = 0.0
rule_reason = ""

print(f"\n  Live V5 — Generalised + Phone-Aware ({len(FEATURE_COLS)} features)")
print("    q=Quit  m=Toggle XGB/RF  y=Toggle YOLO  d=Toggle debug\n")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    fh, fw = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    (face_res, hand_res, pose_res,
     features, yolo_dets, phone_boxes, yolo_person) = extract_live_features(frame, rgb)

    buffer.append(features)

    # Object summary
    obj_names = [d[6] for d in yolo_dets if d[4] != YOLO_CLASSES["person"]]
    obj_str = ("Objects: " + ", ".join(sorted(set(obj_names)))) if obj_names else "Objects: none"

    # ── Prediction ────────────────────────────────────────────────────────
    if len(buffer) >= 3:
        avg_feat = np.mean(buffer, axis=0).reshape(1, -1)
        model = xgb_model if use_xgb else rf_model
        pred = model.predict(avg_feat)[0]
        proba = model.predict_proba(avg_feat)[0]
        confidence = proba[pred]
        ml_label = le.inverse_transform([pred])[0]

        # ── Phone override: YOLO phone detected + holding signals ─────
        phone_signals = []
        if len(phone_boxes) > 0:
            phone_signals.append("YOLO phone")
            # Check head/gaze alignment
            if features[FEATURE_COLS.index("head_towards_phone")] > 0.5:
                phone_signals.append("head towards")
            if features[FEATURE_COLS.index("gaze_phone_align")] > 0.1:
                phone_signals.append("gaze aligned")
            if features[FEATURE_COLS.index("wrist_near_phone")] > 0.5:
                phone_signals.append("wrist near")
            if features[FEATURE_COLS.index("phone_holding_score")] > 0.4:
                phone_signals.append(f"score={features[FEATURE_COLS.index('phone_holding_score')]:.2f}")

        # Rule-based phone detection (even without YOLO phone)
        face_lm = face_res.multi_face_landmarks[0].landmark if (
            face_res.multi_face_landmarks and len(face_res.multi_face_landmarks) > 0
        ) else None
        pose_lm_obj = pose_res.pose_landmarks

        if not phone_signals:
            rule_phone, rule_reason_str = rule_phone_without_yolo(
                face_lm, pose_lm_obj, fh, fw
            )
            if rule_phone:
                phone_signals.append("Rule: " + rule_reason_str)

        # Temporal smoothing: require sustained phone signals
        if phone_signals:
            phone_hold_frames += 1
        else:
            phone_hold_frames = max(0, phone_hold_frames - 2)

        if phone_hold_frames >= PHONE_HOLD_THRESHOLD:
            current_label = "using phone"
            confidence = min(0.95, confidence + 0.2)
            rule_reason = " | ".join(phone_signals)
        else:
            current_label = ml_label
            rule_reason = f"ML prediction (conf={confidence:.2f})"

            # If no person detected by any method
            if not yolo_person and not (pose_res.pose_landmarks is not None) and not face_detected:
                current_label = "no person"
                rule_reason = "No person detected"
    else:
        current_label = "Detecting..."
        rule_reason = "Buffering frames..."

    face_detected = face_res.multi_face_landmarks is not None and len(face_res.multi_face_landmarks) > 0

    # ── Draw overlays ─────────────────────────────────────────────────────
    if show_yolo:
        draw_yolo_boxes(frame, yolo_dets)

    if face_detected:
        for fl in face_res.multi_face_landmarks:
            mp_drawing.draw_landmarks(
                frame, fl, mp_face_mesh.FACEMESH_TESSELATION, None,
                mp_drawing_styles.get_default_face_mesh_tesselation_style()
            )

    if hand_res.multi_hand_landmarks:
        for hl in hand_res.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame, hl, mp_hands.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=1),
            )

    if pose_res.pose_landmarks:
        mp_drawing.draw_landmarks(
            frame, pose_res.pose_landmarks, mp_pose.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
            mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=1),
        )

    # ── UI panel ──────────────────────────────────────────────────────────
    color = LABEL_COLORS.get(current_label, (255, 255, 255))
    model_name = "XGBoost" if use_xgb else "Random Forest"

    panel_h = 170 if show_debug else 130
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (600, panel_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

    cv2.putText(frame, current_label.upper(), (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)

    cv2.putText(frame, f"Conf: {confidence*100:.1f}%  |  {model_name} (V5 Generalised)",
                (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (200, 200, 200), 1)

    cv2.putText(frame, obj_str, (20, 105),
                cv2.FONT_HERSHEY_SIMPLEX, 0.50, (100, 220, 255), 1)

    if show_debug:
        cv2.putText(frame, f"Reason: {rule_reason[:80]}", (20, 130),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, (150, 255, 150), 1)

        debug_parts = []
        if face_detected:
            p_d = features[FEATURE_COLS.index("head_pitch_deg")]
            y_d = features[FEATURE_COLS.index("head_yaw_deg")]
            debug_parts.append(f"Pitch={p_d:.0f}°")
            debug_parts.append(f"Yaw={y_d:.0f}°")
        ph_score = features[FEATURE_COLS.index("phone_holding_score")]
        debug_parts.append(f"PhScore={ph_score:.2f}")
        debug_parts.append(f"PhFrames={phone_hold_frames}")
        debug_parts.append(f"Phones={len(phone_boxes)}")
        cv2.putText(frame, "  ".join(debug_parts), (20, 155),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200, 200, 100), 1)

    cv2.putText(frame, "q:Quit m:Model y:YOLO d:Debug", (fw - 280, fh - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (150, 150, 150), 1)

    cv2.imshow("Behaviour Detection V5 — Generalised", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('m'):
        use_xgb = not use_xgb
        buffer.clear()
        phone_hold_frames = 0
        current_label = "Switching..."
        print(f"  Switched to {'XGBoost' if use_xgb else 'Random Forest'}")
    elif key == ord('y'):
        show_yolo = not show_yolo
        print(f"  YOLO overlay: {'ON' if show_yolo else 'OFF'}")
    elif key == ord('d'):
        show_debug = not show_debug
        print(f"  Debug info: {'ON' if show_debug else 'OFF'}")

cap.release()
cv2.destroyAllWindows()
print("Done.")
