"""
V3 feature extraction: MediaPipe (39) + image-level (24) + YOLO object detection (18) = 81 features.
YOLO detects objects like phone, book, pen, laptop, bottle, cup etc. to better
identify person behaviour (studying, distracted, using phone, etc.).
"""

import os
import sys
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
from tqdm import tqdm
from ultralytics import YOLO

# ─── Configuration ────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(BASE_DIR, "data")
OUTPUT_FILE = os.path.join(BASE_DIR, "features", "m_dataset_features_v3.csv")

# ─── YOLO setup ───────────────────────────────────────────────────────────────
# YOLOv8n is the smallest / fastest model — good enough for object presence
yolo_model = YOLO("yolov8n.pt")  # auto-downloads on first run (~6 MB)

# COCO class IDs we care about for behaviour analysis
YOLO_CLASSES = {
    "person":    0,
    "phone":     67,   # cell phone
    "book":      73,
    "laptop":    63,
    "remote":    65,
    "mouse":     64,
    "keyboard":  66,
    "tv":        62,   # tv / monitor
    "bottle":    39,
    "cup":       41,
    "scissors":  76,
    "clock":     74,
    "backpack":  24,
    "handbag":   26,
}

# Objects considered distractors vs study-related
DISTRACTOR_IDS = {67, 65, 62, 74}          # phone, remote, tv, clock
STUDY_IDS      = {73, 63, 64, 66, 76}      # book, laptop, mouse, keyboard, scissors(pen-like)

# ─── MediaPipe ────────────────────────────────────────────────────────────────
mp_face_mesh = mp.solutions.face_mesh
mp_hands = mp.solutions.hands
mp_pose = mp.solutions.pose

face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=True, max_num_faces=1,
    refine_landmarks=True, min_detection_confidence=0.3,
)
hands = mp_hands.Hands(
    static_image_mode=True, max_num_hands=2,
    min_detection_confidence=0.3,
)
pose = mp_pose.Pose(
    static_image_mode=True, model_complexity=1,
    min_detection_confidence=0.3,
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


# ─── Image-level features (24 features — identical to V2) ────────────────────

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
    """
    Run YOLOv8 on the image and extract behaviour-relevant object features.
    Returns a dict with 18 YOLO-based features.
    """
    h, w = image.shape[:2]
    img_area = h * w

    feat = {
        # Person detection
        "yolo_person_detected": 0,
        "yolo_person_conf": 0.0,
        "yolo_person_area": 0.0,
        # Phone detection
        "yolo_phone_detected": 0,
        "yolo_phone_conf": 0.0,
        "yolo_phone_area": 0.0,
        "yolo_phone_center_x": 0.0,
        "yolo_phone_center_y": 0.0,
        # Book detection
        "yolo_book_detected": 0,
        "yolo_book_conf": 0.0,
        "yolo_book_area": 0.0,
        # Laptop detection
        "yolo_laptop_detected": 0,
        "yolo_laptop_conf": 0.0,
        "yolo_laptop_area": 0.0,
        # Aggregate counts
        "yolo_num_objects": 0,
        "yolo_num_distractors": 0,
        "yolo_num_study_items": 0,
        # Bottles / cups (drinking → distracted proxy)
        "yolo_bottle_or_cup": 0,
    }

    # Run YOLO inference (suppress verbose output)
    results = yolo_model(image, verbose=False, conf=0.25)

    if len(results) == 0 or results[0].boxes is None or len(results[0].boxes) == 0:
        return feat

    boxes = results[0].boxes
    cls_ids = boxes.cls.cpu().numpy().astype(int)
    confs = boxes.conf.cpu().numpy()
    xyxy = boxes.xyxy.cpu().numpy()  # x1, y1, x2, y2

    feat["yolo_num_objects"] = len(cls_ids)

    for i, cid in enumerate(cls_ids):
        x1, y1, x2, y2 = xyxy[i]
        box_area = (x2 - x1) * (y2 - y1) / img_area  # normalized
        cx = ((x1 + x2) / 2.0) / w
        cy = ((y1 + y2) / 2.0) / h
        conf = float(confs[i])

        # Person
        if cid == YOLO_CLASSES["person"]:
            if conf > feat["yolo_person_conf"]:
                feat["yolo_person_detected"] = 1
                feat["yolo_person_conf"] = conf
                feat["yolo_person_area"] = box_area

        # Phone (cell phone)
        elif cid == YOLO_CLASSES["phone"]:
            if conf > feat["yolo_phone_conf"]:
                feat["yolo_phone_detected"] = 1
                feat["yolo_phone_conf"] = conf
                feat["yolo_phone_area"] = box_area
                feat["yolo_phone_center_x"] = cx
                feat["yolo_phone_center_y"] = cy

        # Book
        elif cid == YOLO_CLASSES["book"]:
            if conf > feat["yolo_book_conf"]:
                feat["yolo_book_detected"] = 1
                feat["yolo_book_conf"] = conf
                feat["yolo_book_area"] = box_area

        # Laptop
        elif cid == YOLO_CLASSES["laptop"]:
            if conf > feat["yolo_laptop_conf"]:
                feat["yolo_laptop_detected"] = 1
                feat["yolo_laptop_conf"] = conf
                feat["yolo_laptop_area"] = box_area

        # Bottle or cup
        if cid in (YOLO_CLASSES["bottle"], YOLO_CLASSES["cup"]):
            feat["yolo_bottle_or_cup"] = 1

        # Aggregate distractor / study item counts
        if cid in DISTRACTOR_IDS:
            feat["yolo_num_distractors"] += 1
        if cid in STUDY_IDS:
            feat["yolo_num_study_items"] += 1

    return feat


# ─── Main feature extraction ─────────────────────────────────────────────────

def extract_features(image_path, label):
    image = cv2.imread(image_path)
    if image is None:
        return None

    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # ── Image-level features (always available) ──
    feat = {"label": label}
    img_feat = compute_image_features(image)
    feat.update(img_feat)

    # ── YOLO object-detection features ──
    yolo_feat = compute_yolo_features(image)
    feat.update(yolo_feat)

    # ── MediaPipe detections ──
    face_results = face_mesh.process(rgb)
    hand_results = hands.process(rgb)
    pose_results = pose.process(rgb)

    face_detected = (face_results.multi_face_landmarks is not None
                     and len(face_results.multi_face_landmarks) > 0)
    feat["face_detected"] = int(face_detected)

    num_hands = len(hand_results.multi_hand_landmarks) if hand_results.multi_hand_landmarks else 0
    hand_near_face = 0
    hand_near_ear = 0
    avg_hand_y = 0.0

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

        lb_y = np.mean([lm[i].y for i in LEFT_EYEBROW])
        rb_y = np.mean([lm[i].y for i in RIGHT_EYEBROW])
        le_y = np.mean([lm[i].y for i in LEFT_EYE])
        re_y = np.mean([lm[i].y for i in RIGHT_EYE])
        feat["eyebrow_left_dist"] = le_y - lb_y
        feat["eyebrow_right_dist"] = re_y - rb_y

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

        # Hand-face relationships
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
    else:
        for k in ["ear_left", "ear_right", "ear_avg", "mar",
                   "head_pitch", "head_yaw", "head_roll",
                   "eyebrow_left_dist", "eyebrow_right_dist",
                   "gaze_x", "gaze_y",
                   "face_x_min", "face_x_max", "face_y_min", "face_y_max",
                   "face_width", "face_height", "nose_x", "nose_y"]:
            feat[k] = 0.0

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
    else:
        feat["hand_spread_avg"] = 0.0
        feat["hand1_wrist_x"] = 0.0
        feat["hand1_wrist_y"] = 0.0

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
        le = _pt_pose(plm, mp_pose.PoseLandmark.LEFT_ELBOW)
        re = _pt_pose(plm, mp_pose.PoseLandmark.RIGHT_ELBOW)
        feat["left_elbow_y"] = le[1]
        feat["right_elbow_y"] = re[1]
        lw = _pt_pose(plm, mp_pose.PoseLandmark.LEFT_WRIST)
        rw = _pt_pose(plm, mp_pose.PoseLandmark.RIGHT_WRIST)
        feat["left_wrist_y"] = lw[1]
        feat["right_wrist_y"] = rw[1]
        feat["left_arm_raised"] = float(lw[1] < ls[1])
        feat["right_arm_raised"] = float(rw[1] < rs[1])
    else:
        for k in ["shoulder_width", "shoulder_mid_x", "shoulder_mid_y",
                   "shoulder_roll", "head_shoulder_dist",
                   "left_elbow_y", "right_elbow_y",
                   "left_wrist_y", "right_wrist_y",
                   "left_arm_raised", "right_arm_raised"]:
            feat[k] = 0.0

    return feat


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    if not os.path.isdir(DATASET_DIR):
        print(f"ERROR: {DATASET_DIR} not found")
        sys.exit(1)

    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

    classes = sorted(d for d in os.listdir(DATASET_DIR)
                     if os.path.isdir(os.path.join(DATASET_DIR, d)))
    print(f"Classes: {classes}")

    all_features = []
    total_skipped = 0

    for label in classes:
        class_path = os.path.join(DATASET_DIR, label)
        images = [f for f in os.listdir(class_path)
                  if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp"))]
        print(f"\nProcessing '{label}' — {len(images)} images")

        skipped = 0
        for img_name in tqdm(images, desc=label):
            img_path = os.path.join(class_path, img_name)
            feat = extract_features(img_path, label)
            if feat is not None:
                feat["filename"] = img_name
                all_features.append(feat)
            else:
                skipped += 1
        total_skipped += skipped
        print(f"  Skipped: {skipped}")

    df = pd.DataFrame(all_features)
    cols = ["label", "filename"] + [c for c in df.columns if c not in ("label", "filename")]
    df = df[cols]
    df.to_csv(OUTPUT_FILE, index=False)

    print(f"\nDone! Saved {len(df)} rows × {len(df.columns)-2} features to {OUTPUT_FILE}")
    print(f"Skipped: {total_skipped}")
    feat_cols = [c for c in df.columns if c not in ("label", "filename")]
    print(f"\nFeatures ({len(feat_cols)}):")
    for c in feat_cols:
        print(f"  - {c}")


if __name__ == "__main__":
    main()
