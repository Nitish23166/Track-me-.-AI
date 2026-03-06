"""
Live webcam test for behaviour detection — V3 (81 features).
Uses MediaPipe (39) + image-level stats (24) + YOLO object detection (18).
YOLO detects phone, book, laptop, pen, bottle etc. and displays bounding boxes.
Press 'q' to quit, 'm' to toggle XGBoost / Random Forest, 'y' to toggle YOLO overlay.
"""

import os
import sys
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

xgb_path = os.path.join(MODELS_DIR, "xgb_behaviour_model_v3.pkl")
rf_path = os.path.join(MODELS_DIR, "rf_behaviour_model_v3.pkl")
le_path = os.path.join(MODELS_DIR, "label_encoder_v3.pkl")

# ─── Load models ──────────────────────────────────────────────────────────────
print("Loading V3 models (with YOLO features)...")
xgb_model = joblib.load(xgb_path)
rf_model = joblib.load(rf_path)
le = joblib.load(le_path)
print(f"  Classes: {list(le.classes_)}")

use_xgb = True
show_yolo = True  # toggle YOLO bounding-box overlay

# ─── YOLO setup ───────────────────────────────────────────────────────────────
print("Loading YOLOv8n model...")
yolo_model = YOLO("yolov8n.pt")

# COCO class IDs relevant for behaviour analysis
YOLO_CLASSES = {
    "person":    0,
    "phone":     67,
    "book":      73,
    "laptop":    63,
    "remote":    65,
    "mouse":     64,
    "keyboard":  66,
    "tv":        62,
    "bottle":    39,
    "cup":       41,
    "scissors":  76,
    "clock":     74,
    "backpack":  24,
    "handbag":   26,
}
YOLO_ID_TO_NAME = {v: k for k, v in YOLO_CLASSES.items()}

DISTRACTOR_IDS = {67, 65, 62, 74}
STUDY_IDS = {73, 63, 64, 66, 76}

# Colours for YOLO bounding boxes by category
YOLO_BOX_COLORS = {
    "distractor": (0, 0, 255),    # red
    "study":      (0, 200, 0),    # green
    "person":     (255, 200, 0),  # cyan
    "other":      (200, 200, 0),  # teal
}

# ─── MediaPipe init ───────────────────────────────────────────────────────────
mp_face_mesh = mp.solutions.face_mesh
mp_hands = mp.solutions.hands
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False, max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.3, min_tracking_confidence=0.4,
)
hands = mp_hands.Hands(
    static_image_mode=False, max_num_hands=2,
    min_detection_confidence=0.3, min_tracking_confidence=0.4,
)
pose = mp_pose.Pose(
    static_image_mode=False, model_complexity=1,
    min_detection_confidence=0.3, min_tracking_confidence=0.4,
)

# ─── Landmark indices ─────────────────────────────────────────────────────────
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
LEFT_EYEBROW = [70, 63, 105, 66, 107]
RIGHT_EYEBROW = [300, 293, 334, 296, 336]
FACE_OVAL = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323,
             361, 288, 397, 365, 379, 378, 400, 377, 152, 148,
             176, 149, 150, 136, 172, 58, 132, 93, 234, 127,
             162, 21, 54, 103, 67, 109]

# V3 feature columns (81 total) — must match training
FEATURE_COLS = [
    # 24 image-level features
    "img_brightness", "img_contrast",
    "img_top_brightness", "img_mid_brightness", "img_bot_brightness",
    "img_saturation_mean", "img_saturation_std",
    "img_skin_ratio", "img_skin_upper", "img_skin_lower",
    "img_edge_density", "img_edge_center",
    "img_gradient_mean", "img_gradient_std",
    "img_b_mean", "img_b_std", "img_g_mean", "img_g_std", "img_r_mean", "img_r_std",
    "img_bright_region_upper",
    "img_phone_rect_count", "img_phone_rect_max_area",
    "img_texture_var",
    # 18 YOLO object-detection features
    "yolo_person_detected", "yolo_person_conf", "yolo_person_area",
    "yolo_phone_detected", "yolo_phone_conf", "yolo_phone_area",
    "yolo_phone_center_x", "yolo_phone_center_y",
    "yolo_book_detected", "yolo_book_conf", "yolo_book_area",
    "yolo_laptop_detected", "yolo_laptop_conf", "yolo_laptop_area",
    "yolo_num_objects", "yolo_num_distractors", "yolo_num_study_items",
    "yolo_bottle_or_cup",
    # 39 MediaPipe features
    "face_detected", "ear_left", "ear_right", "ear_avg", "mar",
    "head_pitch", "head_yaw", "head_roll",
    "eyebrow_left_dist", "eyebrow_right_dist",
    "gaze_x", "gaze_y",
    "face_x_min", "face_x_max", "face_y_min", "face_y_max",
    "face_width", "face_height", "nose_x", "nose_y",
    "num_hands", "hand_near_face", "hand_near_ear", "avg_hand_y",
    "hand_spread_avg", "hand1_wrist_x", "hand1_wrist_y",
    "pose_detected", "shoulder_width", "shoulder_mid_x", "shoulder_mid_y",
    "shoulder_roll", "head_shoulder_dist",
    "left_elbow_y", "right_elbow_y",
    "left_wrist_y", "right_wrist_y",
    "left_arm_raised", "right_arm_raised",
]


# ─── Helper functions ─────────────────────────────────────────────────────────

def _pt(lm, idx):
    return np.array([lm[idx].x, lm[idx].y, lm[idx].z])


def _pt_pose(landmarks, idx):
    lm = landmarks[idx.value] if hasattr(idx, "value") else landmarks[idx]
    return np.array([lm.x, lm.y, lm.z])


def compute_ear(landmarks, eye_indices):
    p1 = _pt(landmarks, eye_indices[0])[:2]
    p2 = _pt(landmarks, eye_indices[1])[:2]
    p3 = _pt(landmarks, eye_indices[2])[:2]
    p4 = _pt(landmarks, eye_indices[3])[:2]
    p5 = _pt(landmarks, eye_indices[4])[:2]
    p6 = _pt(landmarks, eye_indices[5])[:2]
    vertical = np.linalg.norm(p2 - p6) + np.linalg.norm(p3 - p5)
    horizontal = np.linalg.norm(p1 - p4)
    return vertical / (2.0 * horizontal) if horizontal > 0 else 0.0


def compute_mar(landmarks):
    top = _pt(landmarks, 13)[:2]
    bottom = _pt(landmarks, 14)[:2]
    left = _pt(landmarks, 61)[:2]
    right = _pt(landmarks, 291)[:2]
    v = np.linalg.norm(top - bottom)
    h = np.linalg.norm(left - right)
    return v / h if h > 0 else 0.0


def compute_head_pose(landmarks):
    nose = _pt(landmarks, 1)
    chin = _pt(landmarks, 152)
    le = _pt(landmarks, 33)
    re = _pt(landmarks, 263)
    eye_mid = (le + re) / 2.0
    ed = np.linalg.norm(le[:2] - re[:2])
    if ed == 0:
        return 0.0, 0.0, 0.0
    yaw = (nose[0] - eye_mid[0]) / ed
    fh = np.linalg.norm(eye_mid[:2] - chin[:2])
    pitch = (nose[1] - eye_mid[1]) / fh if fh > 0 else 0.0
    roll = np.degrees(np.arctan2(re[1] - le[1], re[0] - le[0]))
    return pitch, yaw, roll


def compute_gaze(landmarks):
    try:
        li = _pt(landmarks, 468)[:2]
        ri = _pt(landmarks, 473)[:2]
        lc = (_pt(landmarks, 133)[:2] + _pt(landmarks, 33)[:2]) / 2.0
        rc = (_pt(landmarks, 362)[:2] + _pt(landmarks, 263)[:2]) / 2.0
        lw = np.linalg.norm(_pt(landmarks, 133)[:2] - _pt(landmarks, 33)[:2])
        rw = np.linalg.norm(_pt(landmarks, 362)[:2] - _pt(landmarks, 263)[:2])
        if lw == 0 or rw == 0:
            return 0.0, 0.0
        g = ((li - lc) / lw + (ri - rc) / rw) / 2.0
        return float(g[0]), float(g[1])
    except Exception:
        return 0.0, 0.0


def compute_eyebrow_features(landmarks):
    left_brow_y = np.mean([landmarks[i].y for i in LEFT_EYEBROW])
    right_brow_y = np.mean([landmarks[i].y for i in RIGHT_EYEBROW])
    left_eye_y = np.mean([landmarks[i].y for i in LEFT_EYE])
    right_eye_y = np.mean([landmarks[i].y for i in RIGHT_EYE])
    return left_eye_y - left_brow_y, right_eye_y - right_brow_y


# ─── Image-level features (24 features) ──────────────────────────────────────

def compute_image_features(image):
    h, w = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    feat = {}

    feat["img_brightness"] = gray.mean() / 255.0
    feat["img_contrast"] = gray.std() / 255.0

    h3 = h // 3
    feat["img_top_brightness"] = gray[:h3, :].mean() / 255.0
    feat["img_mid_brightness"] = gray[h3:2*h3, :].mean() / 255.0
    feat["img_bot_brightness"] = gray[2*h3:, :].mean() / 255.0

    sat = hsv[:, :, 1]
    feat["img_saturation_mean"] = sat.mean() / 255.0
    feat["img_saturation_std"] = sat.std() / 255.0

    lower_skin = np.array([0, 20, 70])
    upper_skin = np.array([20, 255, 255])
    skin_mask = cv2.inRange(hsv, lower_skin, upper_skin)
    feat["img_skin_ratio"] = skin_mask.mean() / 255.0
    feat["img_skin_upper"] = skin_mask[:h//2, :].mean() / 255.0
    feat["img_skin_lower"] = skin_mask[h//2:, :].mean() / 255.0

    edges = cv2.Canny(gray, 50, 150)
    feat["img_edge_density"] = edges.mean() / 255.0
    cw1, cw2 = w // 4, 3 * w // 4
    ch1, ch2 = h // 4, 3 * h // 4
    feat["img_edge_center"] = edges[ch1:ch2, cw1:cw2].mean() / 255.0

    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    mag = np.sqrt(gx**2 + gy**2)
    feat["img_gradient_mean"] = mag.mean() / 255.0
    feat["img_gradient_std"] = mag.std() / 255.0

    for i, ch_name in enumerate(["b", "g", "r"]):
        ch = image[:, :, i].astype(float) / 255.0
        feat[f"img_{ch_name}_mean"] = ch.mean()
        feat[f"img_{ch_name}_std"] = ch.std()

    _, bright = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)
    bright_upper = bright[:2*h3, :]
    feat["img_bright_region_upper"] = bright_upper.mean() / 255.0

    contours, _ = cv2.findContours(bright_upper, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    rect_count = 0
    max_rect_area = 0.0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 500:
            x, y, cw_r, ch_r = cv2.boundingRect(cnt)
            aspect = cw_r / ch_r if ch_r > 0 else 0
            if 0.3 <= aspect <= 0.85 or 1.2 <= aspect <= 2.5:
                rect_count += 1
                max_rect_area = max(max_rect_area, area / (h * w))
    feat["img_phone_rect_count"] = rect_count
    feat["img_phone_rect_max_area"] = max_rect_area

    feat["img_texture_var"] = cv2.Laplacian(gray, cv2.CV_64F).var() / 10000.0

    return feat


# ─── YOLO object-detection features (18 features) ────────────────────────────

def compute_yolo_features(image):
    """Run YOLOv8 and extract 18 behaviour-relevant object features.
       Also returns raw detection results for drawing bounding boxes."""
    h, w = image.shape[:2]
    img_area = h * w

    feat = {
        "yolo_person_detected": 0, "yolo_person_conf": 0.0, "yolo_person_area": 0.0,
        "yolo_phone_detected": 0, "yolo_phone_conf": 0.0, "yolo_phone_area": 0.0,
        "yolo_phone_center_x": 0.0, "yolo_phone_center_y": 0.0,
        "yolo_book_detected": 0, "yolo_book_conf": 0.0, "yolo_book_area": 0.0,
        "yolo_laptop_detected": 0, "yolo_laptop_conf": 0.0, "yolo_laptop_area": 0.0,
        "yolo_num_objects": 0, "yolo_num_distractors": 0, "yolo_num_study_items": 0,
        "yolo_bottle_or_cup": 0,
    }

    detections = []  # (x1,y1,x2,y2, class_id, conf, name)

    results = yolo_model(image, verbose=False, conf=0.25)

    if len(results) == 0 or results[0].boxes is None or len(results[0].boxes) == 0:
        return feat, detections

    boxes = results[0].boxes
    cls_ids = boxes.cls.cpu().numpy().astype(int)
    confs = boxes.conf.cpu().numpy()
    xyxy = boxes.xyxy.cpu().numpy()

    feat["yolo_num_objects"] = len(cls_ids)

    for i, cid in enumerate(cls_ids):
        x1, y1, x2, y2 = xyxy[i]
        box_area = (x2 - x1) * (y2 - y1) / img_area
        cx = ((x1 + x2) / 2.0) / w
        cy = ((y1 + y2) / 2.0) / h
        conf = float(confs[i])

        # Store detection for drawing
        name = YOLO_ID_TO_NAME.get(cid, None)
        if name is not None:
            detections.append((int(x1), int(y1), int(x2), int(y2), cid, conf, name))

        if cid == YOLO_CLASSES["person"]:
            if conf > feat["yolo_person_conf"]:
                feat["yolo_person_detected"] = 1
                feat["yolo_person_conf"] = conf
                feat["yolo_person_area"] = box_area

        elif cid == YOLO_CLASSES["phone"]:
            if conf > feat["yolo_phone_conf"]:
                feat["yolo_phone_detected"] = 1
                feat["yolo_phone_conf"] = conf
                feat["yolo_phone_area"] = box_area
                feat["yolo_phone_center_x"] = cx
                feat["yolo_phone_center_y"] = cy

        elif cid == YOLO_CLASSES["book"]:
            if conf > feat["yolo_book_conf"]:
                feat["yolo_book_detected"] = 1
                feat["yolo_book_conf"] = conf
                feat["yolo_book_area"] = box_area

        elif cid == YOLO_CLASSES["laptop"]:
            if conf > feat["yolo_laptop_conf"]:
                feat["yolo_laptop_detected"] = 1
                feat["yolo_laptop_conf"] = conf
                feat["yolo_laptop_area"] = box_area

        if cid in (YOLO_CLASSES["bottle"], YOLO_CLASSES["cup"]):
            feat["yolo_bottle_or_cup"] = 1

        if cid in DISTRACTOR_IDS:
            feat["yolo_num_distractors"] += 1
        if cid in STUDY_IDS:
            feat["yolo_num_study_items"] += 1

    return feat, detections


# ─── Combined feature extraction ─────────────────────────────────────────────

def extract_live_features(bgr_frame, rgb_frame, face_mesh_det, hands_det, pose_det):
    """Extract all 81 V3 features: 24 image + 18 YOLO + 39 MediaPipe.
       Returns (face_results, hand_results, pose_results, feature_vec, yolo_detections)."""
    feat = {k: 0.0 for k in FEATURE_COLS}

    # --- 24 image-level features ---
    img_feats = compute_image_features(bgr_frame)
    feat.update(img_feats)

    # --- 18 YOLO features ---
    yolo_feats, yolo_dets = compute_yolo_features(bgr_frame)
    feat.update(yolo_feats)

    # --- 39 MediaPipe features ---
    face_results = face_mesh_det.process(rgb_frame)
    hand_results = hands_det.process(rgb_frame)
    pose_results = pose_det.process(rgb_frame)

    face_detected = (face_results.multi_face_landmarks is not None
                     and len(face_results.multi_face_landmarks) > 0)
    feat["face_detected"] = int(face_detected)

    num_hands = 0
    hand_near_face = 0
    hand_near_ear = 0
    avg_hand_y = 0.0

    if hand_results.multi_hand_landmarks:
        num_hands = len(hand_results.multi_hand_landmarks)

    if face_detected:
        lm = face_results.multi_face_landmarks[0].landmark

        feat["ear_left"] = compute_ear(lm, LEFT_EYE)
        feat["ear_right"] = compute_ear(lm, RIGHT_EYE)
        feat["ear_avg"] = (feat["ear_left"] + feat["ear_right"]) / 2.0
        feat["mar"] = compute_mar(lm)

        pitch, yaw, roll = compute_head_pose(lm)
        feat["head_pitch"] = pitch
        feat["head_yaw"] = yaw
        feat["head_roll"] = roll

        eb_l, eb_r = compute_eyebrow_features(lm)
        feat["eyebrow_left_dist"] = eb_l
        feat["eyebrow_right_dist"] = eb_r

        gx, gy = compute_gaze(lm)
        feat["gaze_x"] = gx
        feat["gaze_y"] = gy

        xs = [lm[i].x for i in FACE_OVAL]
        ys = [lm[i].y for i in FACE_OVAL]
        feat["face_x_min"] = min(xs)
        feat["face_x_max"] = max(xs)
        feat["face_y_min"] = min(ys)
        feat["face_y_max"] = max(ys)
        feat["face_width"] = feat["face_x_max"] - feat["face_x_min"]
        feat["face_height"] = feat["face_y_max"] - feat["face_y_min"]
        face_cx = (feat["face_x_min"] + feat["face_x_max"]) / 2.0
        face_cy = (feat["face_y_min"] + feat["face_y_max"]) / 2.0
        feat["nose_x"] = lm[1].x
        feat["nose_y"] = lm[1].y

        if hand_results.multi_hand_landmarks:
            for hand_lm in hand_results.multi_hand_landmarks:
                hx = np.mean([l.x for l in hand_lm.landmark])
                hy = np.mean([l.y for l in hand_lm.landmark])
                dist = np.sqrt((hx - face_cx)**2 + (hy - face_cy)**2)
                if dist < 0.25:
                    hand_near_face += 1
                ear_lx, ear_ly = lm[234].x, lm[234].y
                ear_rx, ear_ry = lm[454].x, lm[454].y
                d_l = np.sqrt((hx - ear_lx)**2 + (hy - ear_ly)**2)
                d_r = np.sqrt((hx - ear_rx)**2 + (hy - ear_ry)**2)
                if min(d_l, d_r) < 0.15:
                    hand_near_ear += 1
                avg_hand_y += hy
            avg_hand_y /= num_hands

    feat["num_hands"] = num_hands
    feat["hand_near_face"] = hand_near_face
    feat["hand_near_ear"] = hand_near_ear
    feat["avg_hand_y"] = avg_hand_y

    if hand_results.multi_hand_landmarks:
        spreads = []
        for hand_lm in hand_results.multi_hand_landmarks:
            pts = np.array([[l.x, l.y] for l in hand_lm.landmark])
            spreads.append(np.std(pts, axis=0).mean())
        feat["hand_spread_avg"] = np.mean(spreads)
        feat["hand1_wrist_x"] = hand_results.multi_hand_landmarks[0].landmark[0].x
        feat["hand1_wrist_y"] = hand_results.multi_hand_landmarks[0].landmark[0].y

    pose_detected = pose_results.pose_landmarks is not None
    feat["pose_detected"] = int(pose_detected)

    if pose_detected:
        plm = pose_results.pose_landmarks.landmark
        ls = _pt_pose(plm, mp_pose.PoseLandmark.LEFT_SHOULDER)
        rs = _pt_pose(plm, mp_pose.PoseLandmark.RIGHT_SHOULDER)
        feat["shoulder_width"] = np.linalg.norm(ls[:2] - rs[:2])
        sm = (ls + rs) / 2.0
        feat["shoulder_mid_x"] = sm[0]
        feat["shoulder_mid_y"] = sm[1]
        feat["shoulder_roll"] = np.degrees(np.arctan2(rs[1] - ls[1], rs[0] - ls[0]))
        nose_p = _pt_pose(plm, mp_pose.PoseLandmark.NOSE)
        feat["head_shoulder_dist"] = nose_p[1] - sm[1]
        le_p = _pt_pose(plm, mp_pose.PoseLandmark.LEFT_ELBOW)
        re_p = _pt_pose(plm, mp_pose.PoseLandmark.RIGHT_ELBOW)
        feat["left_elbow_y"] = le_p[1]
        feat["right_elbow_y"] = re_p[1]
        lw = _pt_pose(plm, mp_pose.PoseLandmark.LEFT_WRIST)
        rw = _pt_pose(plm, mp_pose.PoseLandmark.RIGHT_WRIST)
        feat["left_wrist_y"] = lw[1]
        feat["right_wrist_y"] = rw[1]
        feat["left_arm_raised"] = float(lw[1] < ls[1])
        feat["right_arm_raised"] = float(rw[1] < rs[1])

    return face_results, hand_results, pose_results, np.array([feat[c] for c in FEATURE_COLS]), yolo_dets


# ─── Draw YOLO bounding boxes ────────────────────────────────────────────────

def draw_yolo_boxes(frame, detections):
    """Draw labelled bounding boxes for detected objects."""
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


# ─── Phone detection helpers (head + eyes towards phone) ─────────────────────

GAZE_PHONE_MARGIN  = 0.18
HEAD_YAW_PHONE_THR = 0.25
NOSE_DOWN_PITCH    = -0.10


def gaze_towards_phone(face_lm, phone_boxes, margin=GAZE_PHONE_MARGIN):
    """Check if eye gaze direction points towards any phone bounding box."""
    try:
        li = np.array([face_lm[468].x, face_lm[468].y])
        ri = np.array([face_lm[473].x, face_lm[473].y])
        lc = (np.array([face_lm[133].x, face_lm[133].y]) +
              np.array([face_lm[33].x, face_lm[33].y])) / 2.0
        rc = (np.array([face_lm[362].x, face_lm[362].y]) +
              np.array([face_lm[263].x, face_lm[263].y])) / 2.0
        gaze_offset = ((li - lc) + (ri - rc)) / 2.0
        nose = np.array([face_lm[1].x, face_lm[1].y])
        gaze_point = nose + gaze_offset * 3.0
        for (bx1, by1, bx2, by2) in phone_boxes:
            if (bx1 - margin <= gaze_point[0] <= bx2 + margin and
                by1 - margin <= gaze_point[1] <= by2 + margin):
                return True
    except Exception:
        pass
    return False


def head_towards_phone(face_lm, phone_boxes):
    """Check if head yaw/pitch orientation aligns with a detected phone."""
    nose = np.array([face_lm[1].x, face_lm[1].y])
    le = np.array([face_lm[33].x, face_lm[33].y])
    re = np.array([face_lm[263].x, face_lm[263].y])
    eye_mid = (le + re) / 2.0
    ed = np.linalg.norm(le - re)
    if ed < 0.01:
        return False
    head_dir_x = (nose[0] - eye_mid[0]) / ed
    chin = np.array([face_lm[152].x, face_lm[152].y])
    face_h = np.linalg.norm(eye_mid - chin)
    head_dir_y = (nose[1] - eye_mid[1]) / face_h if face_h > 0 else 0.0
    for (bx1, by1, bx2, by2) in phone_boxes:
        pcx = (bx1 + bx2) / 2.0
        pcy = (by1 + by2) / 2.0
        phone_dx = pcx - nose[0]
        phone_dy = pcy - nose[1]
        yaw_ok = (phone_dx * head_dir_x > 0) or abs(phone_dx) < 0.10
        pitch_ok = (phone_dy > 0 and head_dir_y > 0) or abs(phone_dy) < 0.08
        if yaw_ok and pitch_ok:
            return True
    return False


def get_nose_pitch_v3(face_lm):
    """Compute vertical pitch of nose relative to eye midpoint."""
    nose = np.array([face_lm[1].x, face_lm[1].y, face_lm[1].z])
    le = np.array([face_lm[33].x, face_lm[33].y, face_lm[33].z])
    re = np.array([face_lm[263].x, face_lm[263].y, face_lm[263].z])
    chin = np.array([face_lm[152].x, face_lm[152].y, face_lm[152].z])
    eye_mid = (le + re) / 2.0
    face_h = np.linalg.norm(eye_mid[:2] - chin[:2])
    return (nose[1] - eye_mid[1]) / face_h if face_h > 0 else 0.0


def detect_phone_using(face_res, yolo_dets):
    """Return True if YOLO phone detected AND head+eyes oriented towards it."""
    # Collect normalised phone bounding boxes from YOLO detections
    phone_boxes = []
    for det in yolo_dets:
        x1, y1, x2, y2, cid, conf, name = det
        if cid == YOLO_CLASSES["phone"]:
            # detections are in pixel coords; we don't have frame dims here,
            # so we'll need them passed in.  Use a wrapper instead.
            phone_boxes.append(det)
    return len(phone_boxes) > 0, phone_boxes


def check_phone_using(face_res, yolo_dets, frame_hw):
    """Complete phone-using check: YOLO phone + head/eyes towards phone.
    Returns (is_phone_using: bool, reason: str)."""
    h, w = frame_hw
    # Get normalised phone boxes
    phone_boxes_norm = []
    for det in yolo_dets:
        x1, y1, x2, y2, cid, conf, name = det
        if cid == YOLO_CLASSES["phone"]:
            phone_boxes_norm.append((x1 / w, y1 / h, x2 / w, y2 / h))

    if not phone_boxes_norm:
        return False, ""

    face_detected = (face_res.multi_face_landmarks is not None
                     and len(face_res.multi_face_landmarks) > 0)
    if not face_detected:
        return False, ""

    flm = face_res.multi_face_landmarks[0].landmark
    reasons = ["YOLO phone"]

    h_ok = head_towards_phone(flm, phone_boxes_norm)
    g_ok = gaze_towards_phone(flm, phone_boxes_norm)
    pitch = get_nose_pitch_v3(flm)
    n_down = pitch > NOSE_DOWN_PITCH

    if h_ok:
        reasons.append("head towards phone")
    if g_ok:
        reasons.append("eyes towards phone")
    if n_down:
        reasons.append("nose down")

    # Decision: need at least 2 signals
    signals = sum([h_ok, g_ok, n_down])
    if signals >= 2:
        return True, " + ".join(reasons)
    # Single signal + phone detected — still likely
    if h_ok and g_ok:
        return True, " + ".join(reasons)
    return False, ""


# ─── Colour map for each behaviour class ─────────────────────────────────────
COLORS = {
    "studying":    (0, 200, 0),
    "distracted":  (0, 100, 255),
    "using phone": (0, 0, 255),
    "phone using": (0, 0, 255),
    "no person":   (180, 180, 180),
}

# ─── Main loop ────────────────────────────────────────────────────────────────
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("ERROR: Cannot open webcam")
    sys.exit(1)

buffer = deque(maxlen=7)
current_label = "Detecting..."
confidence = 0.0
detected_objects_str = ""

print(f"\n  Live test running (V3 — {len(FEATURE_COLS)} features incl. YOLO)!")
print("    q  — Quit")
print("    m  — Toggle XGBoost / Random Forest")
print("    y  — Toggle YOLO bounding boxes\n")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_res, hand_res, pose_res, features, yolo_dets = extract_live_features(
        frame, rgb, face_mesh, hands, pose
    )

    buffer.append(features)

    # Build detected-objects summary string
    obj_names = [d[6] for d in yolo_dets if d[4] != YOLO_CLASSES["person"]]
    if obj_names:
        detected_objects_str = "Objects: " + ", ".join(sorted(set(obj_names)))
    else:
        detected_objects_str = "Objects: none"

    # Predict using smoothed features
    if len(buffer) >= 3:
        avg_feat = np.mean(buffer, axis=0).reshape(1, -1)
        model = xgb_model if use_xgb else rf_model
        pred = model.predict(avg_feat)[0]
        proba = model.predict_proba(avg_feat)[0]
        confidence = proba[pred]
        current_label = le.inverse_transform([pred])[0]

        # ── Phone-using override: YOLO phone + head/eyes towards phone ──
        fh_frame, fw_frame = frame.shape[:2]
        phone_detected, phone_reason = check_phone_using(
            face_res, yolo_dets, (fh_frame, fw_frame)
        )
        if phone_detected:
            current_label = "phone using"
            confidence = 0.95
            detected_objects_str += f"  [{phone_reason}]"

    fh, fw, _ = frame.shape
    color = COLORS.get(current_label, (255, 255, 255))
    model_name = "XGBoost" if use_xgb else "Random Forest"

    # Draw YOLO bounding boxes
    if show_yolo:
        draw_yolo_boxes(frame, yolo_dets)

    # Draw face mesh
    if face_res.multi_face_landmarks:
        for fl in face_res.multi_face_landmarks:
            mp_drawing.draw_landmarks(
                frame, fl,
                mp_face_mesh.FACEMESH_TESSELATION,
                None,
                mp_drawing_styles.get_default_face_mesh_tesselation_style()
            )

    # Draw hand landmarks
    if hand_res.multi_hand_landmarks:
        for hl in hand_res.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame, hl,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=1),
            )

    # Draw pose skeleton
    if pose_res.pose_landmarks:
        mp_drawing.draw_landmarks(
            frame, pose_res.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
            mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=1),
        )

    # ── Overlay UI ──
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (500, 140), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    cv2.putText(frame, current_label.upper(), (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)

    cv2.putText(frame, f"Conf: {confidence*100:.1f}%  |  {model_name} (V3+YOLO)", (20, 85),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

    cv2.putText(frame, detected_objects_str, (20, 120),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (100, 220, 255), 1)

    cv2.putText(frame, "q: Quit | m: Switch model | y: Toggle YOLO", (fw - 380, fh - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 150), 1)

    cv2.imshow("Behaviour Detection V3 (YOLO) - Live", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('m'):
        use_xgb = not use_xgb
        buffer.clear()
        current_label = "Switching..."
        print(f"  Switched to {'XGBoost' if use_xgb else 'Random Forest'}")
    elif key == ord('y'):
        show_yolo = not show_yolo
        print(f"  YOLO overlay: {'ON' if show_yolo else 'OFF'}")

cap.release()
cv2.destroyAllWindows()
print("Done.")
