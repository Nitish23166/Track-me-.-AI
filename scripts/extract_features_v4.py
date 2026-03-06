"""
V4 feature extraction — Generalised & Phone-Aware.

Design goals:
  • Work from ANY camera angle, distance, environment.
  • Replace absolute coordinates with **ratios, angles, and normalised distances**.
  • Use solvePnP for accurate 3D head-pose (pitch/yaw/roll in degrees).
  • Add dedicated phone-usage features (arm angle, hand-face angle, YOLO phone).
  • Keep YOLO object features for context.

Feature groups:
  A. Face geometry        (16)  — EAR, MAR, head-pose (solvePnP + geom), gaze, brow, aspect-ratio
  B. Hand–face relations  (14)  — distances & angles normalised by face/body size
  C. Pose / body          (16)  — angular measurements, relative positions
  D. Phone-specific       (10)  — YOLO phone + spatial relationships
  E. YOLO objects         (14)  — presence, confidence, counts
  F. Image-level          ( 8)  — minimal invariant stats (edge, texture, skin)
  Total: ~78 features
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
OUTPUT_FILE = os.path.join(BASE_DIR, "features", "m_dataset_features_v4.csv")

# ─── YOLO ─────────────────────────────────────────────────────────────────────
yolo_model = YOLO("yolov8n.pt")

YOLO_CLASSES = {
    "person": 0, "phone": 67, "book": 73, "laptop": 63,
    "remote": 65, "mouse": 64, "keyboard": 66, "tv": 62,
    "bottle": 39, "cup": 41, "scissors": 76, "clock": 74,
    "backpack": 24, "handbag": 26,
}
DISTRACTOR_IDS = {67, 65, 62, 74}
STUDY_IDS = {73, 63, 64, 66, 76}

# ─── MediaPipe ────────────────────────────────────────────────────────────────
mp_face_mesh = mp.solutions.face_mesh
mp_hands = mp.solutions.hands
mp_pose = mp.solutions.pose

face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=True, max_num_faces=1,
    refine_landmarks=True, min_detection_confidence=0.3,
)
hands_det = mp_hands.Hands(
    static_image_mode=True, max_num_hands=2,
    min_detection_confidence=0.3,
)
pose_det = mp_pose.Pose(
    static_image_mode=True, model_complexity=2,       # higher accuracy
    min_detection_confidence=0.3,
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

# 3-D model points for solvePnP (canonical face in mm, centred on nose)
MODEL_POINTS_3D = np.array([
    (0.0, 0.0, 0.0),             # Nose tip  (1)
    (0.0, -330.0, -65.0),        # Chin      (152)
    (-225.0, 170.0, -135.0),     # Left eye left corner (33)
    (225.0, 170.0, -135.0),      # Right eye right corner (263)
    (-150.0, -150.0, -125.0),    # Left mouth corner (61)
    (150.0, -150.0, -125.0),     # Right mouth corner (291)
], dtype=np.float64)

FACE_2D_INDICES = [1, 152, 33, 263, 61, 291]

# ═══════════════════════════════════════════════════════════════════════════════
#  HELPER FUNCTIONS
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
    """Angle in degrees between two vectors."""
    cos = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
    return np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))


def _safe_div(a, b, default=0.0):
    return a / b if abs(b) > 1e-8 else default


# ═══════════════════════════════════════════════════════════════════════════════
#  A. FACE GEOMETRY (16 features)
# ═══════════════════════════════════════════════════════════════════════════════

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
    """solvePnP-based head pose → (pitch, yaw, roll) in degrees.
    Camera-angle invariant because it reconstructs 3D orientation."""
    img_pts = np.array(
        [[lm[i].x * w, lm[i].y * h] for i in FACE_2D_INDICES],
        dtype=np.float64,
    )
    focal = w
    cam_matrix = np.array([
        [focal, 0, w / 2],
        [0, focal, h / 2],
        [0, 0, 1],
    ], dtype=np.float64)
    dist_coeffs = np.zeros((4, 1))
    ok, rvec, tvec = cv2.solvePnP(
        MODEL_POINTS_3D, img_pts, cam_matrix, dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not ok:
        return 0.0, 0.0, 0.0
    rmat, _ = cv2.Rodrigues(rvec)
    angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)
    return float(angles[0]), float(angles[1]), float(angles[2])  # pitch, yaw, roll


def compute_head_pose_geom(lm):
    nose, chin = _pt(lm, 1), _pt(lm, 152)
    le, re = _pt(lm, 33), _pt(lm, 263)
    eye_mid = (le + re) / 2.0
    ed = np.linalg.norm(le[:2] - re[:2])
    if ed < 1e-6:
        return 0.0, 0.0, 0.0
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
        if lw < 1e-6 or rw < 1e-6:
            return 0.0, 0.0
        g = ((li - lc) / lw + (ri - rc) / rw) / 2.0
        return float(g[0]), float(g[1])
    except Exception:
        return 0.0, 0.0


def compute_face_features(lm, h, w):
    """Return dict with 16 face-geometry features."""
    f = {}
    # EAR (ratio-based → invariant)
    f["ear_left"] = compute_ear(lm, LEFT_EYE)
    f["ear_right"] = compute_ear(lm, RIGHT_EYE)
    f["ear_avg"] = (f["ear_left"] + f["ear_right"]) / 2.0

    # MAR
    f["mar"] = compute_mar(lm)

    # Head pose via solvePnP (degrees — camera-invariant)
    pitch_s, yaw_s, roll_s = compute_head_pose_solvepnp(lm, h, w)
    f["head_pitch_deg"] = pitch_s
    f["head_yaw_deg"] = yaw_s
    f["head_roll_deg"] = roll_s

    # Geometric head pose (normalised — good secondary signal)
    pitch_g, yaw_g, roll_g = compute_head_pose_geom(lm)
    f["head_pitch_geom"] = pitch_g
    f["head_yaw_geom"] = yaw_g

    # Gaze direction (relative to eye frame — invariant)
    gx, gy = compute_gaze(lm)
    f["gaze_x"] = gx
    f["gaze_y"] = gy

    # Eyebrow distance normalised by face height
    xs = [lm[i].x for i in FACE_OVAL]
    ys = [lm[i].y for i in FACE_OVAL]
    face_w = max(xs) - min(xs)
    face_h = max(ys) - min(ys)
    f["face_aspect_ratio"] = _safe_div(face_w, face_h)
    f["face_area_rel"] = face_w * face_h  # relative to frame (normalised 0-1)

    lb_y = np.mean([lm[i].y for i in LEFT_EYEBROW])
    le_y = np.mean([lm[i].y for i in LEFT_EYE])
    rb_y = np.mean([lm[i].y for i in RIGHT_EYEBROW])
    re_y = np.mean([lm[i].y for i in RIGHT_EYE])
    f["eyebrow_left_dist_norm"] = _safe_div(le_y - lb_y, face_h)
    f["eyebrow_right_dist_norm"] = _safe_div(re_y - rb_y, face_h)

    return f, face_w, face_h


# ═══════════════════════════════════════════════════════════════════════════════
#  B. HAND–FACE RELATIONS (14 features)
# ═══════════════════════════════════════════════════════════════════════════════

def compute_hand_features(face_lm, hand_results, face_w, face_h):
    """Return dict with 14 hand–face features, all normalised."""
    f = {}
    num_hands = 0
    hand_near_face = 0
    hand_near_ear = 0
    hands_in_phone_zone = 0
    hand_face_angles = []
    hand_face_dists = []
    hand_below_face_count = 0
    hand_spread_list = []
    hand_wrist_ys = []

    face_detected = face_lm is not None

    if hand_results.multi_hand_landmarks:
        num_hands = len(hand_results.multi_hand_landmarks)

    if face_detected and hand_results.multi_hand_landmarks:
        face_cx = (min(face_lm[i].x for i in FACE_OVAL) +
                   max(face_lm[i].x for i in FACE_OVAL)) / 2.0
        face_cy = (min(face_lm[i].y for i in FACE_OVAL) +
                   max(face_lm[i].y for i in FACE_OVAL)) / 2.0
        nose = _pt2(face_lm, 1)

        for hand_lm in hand_results.multi_hand_landmarks:
            hx = np.mean([l.x for l in hand_lm.landmark])
            hy = np.mean([l.y for l in hand_lm.landmark])
            wrist = np.array([hand_lm.landmark[0].x, hand_lm.landmark[0].y])
            hand_wrist_ys.append(wrist[1])

            # Distance to face centre (normalised by face diagonal)
            face_diag = np.sqrt(face_w**2 + face_h**2) + 1e-8
            dist = np.sqrt((hx - face_cx)**2 + (hy - face_cy)**2)
            dist_norm = dist / face_diag
            hand_face_dists.append(dist_norm)

            if dist_norm < 1.2:
                hand_near_face += 1

            # Near-ear check
            ear_l = _pt2(face_lm, 234)
            ear_r = _pt2(face_lm, 454)
            d_el = np.linalg.norm(np.array([hx, hy]) - ear_l) / face_diag
            d_er = np.linalg.norm(np.array([hx, hy]) - ear_r) / face_diag
            if min(d_el, d_er) < 0.5:
                hand_near_ear += 1

            # Angle from face centre to hand (0°=right, 90°=down, -90°=up)
            angle = np.degrees(np.arctan2(hy - face_cy, hx - face_cx))
            hand_face_angles.append(angle)

            # Hand below face (phone-holding zone: hand between chin and chest)
            chin_y = face_lm[152].y
            if hy > chin_y and hy < chin_y + face_h * 2.5:
                hands_in_phone_zone += 1
            if hy > face_cy:
                hand_below_face_count += 1

            # Hand spread (open vs. closed)
            pts = np.array([[l.x, l.y] for l in hand_lm.landmark])
            hand_spread_list.append(np.std(pts, axis=0).mean())
    elif not face_detected and hand_results.multi_hand_landmarks:
        for hand_lm in hand_results.multi_hand_landmarks:
            wrist = np.array([hand_lm.landmark[0].x, hand_lm.landmark[0].y])
            hand_wrist_ys.append(wrist[1])
            pts = np.array([[l.x, l.y] for l in hand_lm.landmark])
            hand_spread_list.append(np.std(pts, axis=0).mean())

    f["num_hands"] = num_hands
    f["hand_near_face"] = hand_near_face
    f["hand_near_ear"] = hand_near_ear
    f["hands_in_phone_zone"] = hands_in_phone_zone
    f["hand_below_face"] = hand_below_face_count
    f["hand_face_dist_min"] = min(hand_face_dists) if hand_face_dists else 0.0
    f["hand_face_dist_avg"] = np.mean(hand_face_dists) if hand_face_dists else 0.0
    f["hand_face_angle_avg"] = np.mean(hand_face_angles) if hand_face_angles else 0.0
    f["hand_face_angle_min"] = min(hand_face_angles) if hand_face_angles else 0.0
    f["hand_spread_avg"] = np.mean(hand_spread_list) if hand_spread_list else 0.0
    f["hand_spread_max"] = max(hand_spread_list) if hand_spread_list else 0.0
    # Wrist vertical position relative to frame (low = on desk, mid = phone)
    f["hand_wrist_y_avg"] = np.mean(hand_wrist_ys) if hand_wrist_ys else 0.0
    f["hand_wrist_y_max"] = max(hand_wrist_ys) if hand_wrist_ys else 0.0
    f["hand_wrist_y_min"] = min(hand_wrist_ys) if hand_wrist_ys else 0.0

    return f


# ═══════════════════════════════════════════════════════════════════════════════
#  C. POSE / BODY (16 features — all relative / angular)
# ═══════════════════════════════════════════════════════════════════════════════

def compute_pose_features(pose_landmarks):
    """Return dict with 16 body-pose features (angle/ratio based)."""
    f = {}
    if pose_landmarks is None:
        keys = [
            "pose_detected", "shoulder_roll_deg", "shoulder_width_norm",
            "head_shoulder_dist_norm", "body_lean_angle",
            "left_elbow_angle", "right_elbow_angle",
            "left_wrist_above_shoulder", "right_wrist_above_shoulder",
            "arm_symmetry", "left_wrist_shoulder_dist_norm",
            "right_wrist_shoulder_dist_norm",
            "nose_shoulder_offset_norm", "torso_visible",
            "left_hand_raised_high", "right_hand_raised_high",
        ]
        for k in keys:
            f[k] = 0.0
        return f

    plm = pose_landmarks.landmark

    ls = _pose_pt(plm, mp_pose.PoseLandmark.LEFT_SHOULDER)
    rs = _pose_pt(plm, mp_pose.PoseLandmark.RIGHT_SHOULDER)
    nose = _pose_pt(plm, mp_pose.PoseLandmark.NOSE)
    le = _pose_pt(plm, mp_pose.PoseLandmark.LEFT_ELBOW)
    re = _pose_pt(plm, mp_pose.PoseLandmark.RIGHT_ELBOW)
    lw = _pose_pt(plm, mp_pose.PoseLandmark.LEFT_WRIST)
    rw = _pose_pt(plm, mp_pose.PoseLandmark.RIGHT_WRIST)
    lh = _pose_pt(plm, mp_pose.PoseLandmark.LEFT_HIP)
    rh = _pose_pt(plm, mp_pose.PoseLandmark.RIGHT_HIP)

    shoulder_mid = (ls + rs) / 2.0
    shoulder_w = np.linalg.norm(ls[:2] - rs[:2])

    f["pose_detected"] = 1.0

    # Shoulder roll (degrees)
    f["shoulder_roll_deg"] = np.degrees(np.arctan2(rs[1] - ls[1], rs[0] - ls[0]))

    # Shoulder width normalised by image height proxy (distance shoulder-hip)
    hip_mid = (lh + rh) / 2.0
    torso_h = np.linalg.norm(shoulder_mid[:2] - hip_mid[:2])
    f["shoulder_width_norm"] = _safe_div(shoulder_w, torso_h)

    # Head-to-shoulder distance normalised by shoulder width
    f["head_shoulder_dist_norm"] = _safe_div(
        np.linalg.norm(nose[:2] - shoulder_mid[:2]), shoulder_w
    )

    # Body lean angle (angle of torso midline from vertical)
    torso_vec = hip_mid[:2] - shoulder_mid[:2]
    vertical = np.array([0.0, 1.0])
    f["body_lean_angle"] = _angle_between(torso_vec, vertical) if np.linalg.norm(torso_vec) > 1e-6 else 0.0

    # Elbow angles (angle at elbow between upper-arm and forearm)
    def elbow_angle(shoulder, elbow, wrist):
        v1 = shoulder[:2] - elbow[:2]
        v2 = wrist[:2] - elbow[:2]
        return _angle_between(v1, v2) if (np.linalg.norm(v1) > 1e-6 and np.linalg.norm(v2) > 1e-6) else 180.0

    f["left_elbow_angle"] = elbow_angle(ls, le, lw)
    f["right_elbow_angle"] = elbow_angle(rs, re, rw)

    # Wrist above shoulder (indicator of raised arm / phone holding)
    f["left_wrist_above_shoulder"] = float(lw[1] < ls[1])
    f["right_wrist_above_shoulder"] = float(rw[1] < rs[1])

    # Arm position symmetry (difference in wrist heights normalised)
    f["arm_symmetry"] = _safe_div(abs(lw[1] - rw[1]), shoulder_w)

    # Wrist-to-shoulder distance normalised
    f["left_wrist_shoulder_dist_norm"] = _safe_div(
        np.linalg.norm(lw[:2] - ls[:2]), shoulder_w
    )
    f["right_wrist_shoulder_dist_norm"] = _safe_div(
        np.linalg.norm(rw[:2] - rs[:2]), shoulder_w
    )

    # Nose offset from shoulder midpoint (normalised by shoulder width)
    f["nose_shoulder_offset_norm"] = _safe_div(
        abs(nose[0] - shoulder_mid[0]), shoulder_w
    )

    # Torso visible (both hips detected with reasonable confidence)
    lh_vis = plm[mp_pose.PoseLandmark.LEFT_HIP.value].visibility
    rh_vis = plm[mp_pose.PoseLandmark.RIGHT_HIP.value].visibility
    f["torso_visible"] = float(lh_vis > 0.5 and rh_vis > 0.5)

    # Hands raised high (above nose — waving / stretching)
    f["left_hand_raised_high"] = float(lw[1] < nose[1])
    f["right_hand_raised_high"] = float(rw[1] < nose[1])

    return f


# ═══════════════════════════════════════════════════════════════════════════════
#  D. PHONE-SPECIFIC (10 features)
# ═══════════════════════════════════════════════════════════════════════════════

def compute_phone_features(face_lm, pose_landmarks, phone_boxes, face_w, face_h):
    """
    10 features specifically designed to detect phone usage.
    Uses YOLO phone bbox + face/pose geometry.
    """
    f = {}
    f["phone_detected"] = int(len(phone_boxes) > 0)
    f["phone_count"] = len(phone_boxes)

    best_phone = None
    best_area = 0.0
    for box in phone_boxes:
        bx1, by1, bx2, by2, conf = box
        area = (bx2 - bx1) * (by2 - by1)
        if area > best_area:
            best_area = area
            best_phone = box

    f["phone_box_area"] = best_area

    # Default values
    f["phone_face_dist_norm"] = 0.0
    f["phone_face_angle_deg"] = 0.0
    f["phone_below_face"] = 0.0
    f["gaze_phone_align"] = 0.0
    f["head_towards_phone"] = 0.0
    f["wrist_near_phone"] = 0.0
    f["phone_holding_score"] = 0.0

    if best_phone is None:
        return f

    bx1, by1, bx2, by2, conf = best_phone
    pcx = (bx1 + bx2) / 2.0
    pcy = (by1 + by2) / 2.0

    face_diag = np.sqrt(face_w**2 + face_h**2) + 1e-8

    if face_lm is not None:
        nose = _pt2(face_lm, 1)
        face_cx = (min(face_lm[i].x for i in FACE_OVAL) +
                   max(face_lm[i].x for i in FACE_OVAL)) / 2.0
        face_cy = (min(face_lm[i].y for i in FACE_OVAL) +
                   max(face_lm[i].y for i in FACE_OVAL)) / 2.0

        # Distance from face to phone centre (normalised)
        f["phone_face_dist_norm"] = np.sqrt(
            (face_cx - pcx)**2 + (face_cy - pcy)**2
        ) / face_diag

        # Angle from face centre to phone (negative = phone below face)
        f["phone_face_angle_deg"] = np.degrees(
            np.arctan2(pcy - face_cy, pcx - face_cx)
        )

        # Phone below face?
        f["phone_below_face"] = float(pcy > face_cy)

        # Gaze alignment with phone
        gx, gy = compute_gaze(face_lm)
        gaze_vec = np.array([gx, gy])
        phone_dir = np.array([pcx - nose[0], pcy - nose[1]])
        phone_dir_len = np.linalg.norm(phone_dir)
        if phone_dir_len > 1e-6 and np.linalg.norm(gaze_vec) > 1e-6:
            cos_sim = np.dot(gaze_vec, phone_dir / phone_dir_len)
            f["gaze_phone_align"] = float(cos_sim)

        # Head orientation towards phone (using geometric head yaw/pitch)
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
            f["head_towards_phone"] = float(yaw_ok and pitch_ok)

    # Wrist near phone
    if pose_landmarks is not None:
        plm = pose_landmarks.landmark
        lw = _pose_pt(plm, mp_pose.PoseLandmark.LEFT_WRIST)[:2]
        rw = _pose_pt(plm, mp_pose.PoseLandmark.RIGHT_WRIST)[:2]
        phone_pt = np.array([pcx, pcy])
        sw = np.linalg.norm(
            _pose_pt(plm, mp_pose.PoseLandmark.LEFT_SHOULDER)[:2] -
            _pose_pt(plm, mp_pose.PoseLandmark.RIGHT_SHOULDER)[:2]
        )
        margin = sw * 0.3 if sw > 1e-6 else 0.06
        d_l = np.linalg.norm(lw - phone_pt)
        d_r = np.linalg.norm(rw - phone_pt)
        f["wrist_near_phone"] = float(min(d_l, d_r) < margin)

    # Composite phone-holding score
    signals = (
        f["phone_detected"] * 2.0 +
        f["head_towards_phone"] * 1.5 +
        f["gaze_phone_align"] * 1.0 +
        f["wrist_near_phone"] * 1.5 +
        f["phone_below_face"] * 0.5
    )
    f["phone_holding_score"] = min(signals / 5.0, 1.0)

    return f


# ═══════════════════════════════════════════════════════════════════════════════
#  E. YOLO OBJECTS (14 features)
# ═══════════════════════════════════════════════════════════════════════════════

def compute_yolo_features(image):
    """14 YOLO features + raw phone_boxes list for phone-specific features."""
    h, w = image.shape[:2]
    img_area = h * w

    feat = {
        "yolo_person_detected": 0, "yolo_person_conf": 0.0, "yolo_person_area": 0.0,
        "yolo_phone_conf": 0.0, "yolo_phone_area": 0.0,
        "yolo_book_detected": 0, "yolo_book_area": 0.0,
        "yolo_laptop_detected": 0, "yolo_laptop_area": 0.0,
        "yolo_num_objects": 0, "yolo_num_distractors": 0, "yolo_num_study_items": 0,
        "yolo_bottle_or_cup": 0,
        "yolo_any_distractor": 0,
    }

    phone_boxes = []  # (x1_norm, y1_norm, x2_norm, y2_norm, conf)

    results = yolo_model(image, verbose=False, conf=0.25)
    if len(results) == 0 or results[0].boxes is None or len(results[0].boxes) == 0:
        return feat, phone_boxes

    boxes = results[0].boxes
    cls_ids = boxes.cls.cpu().numpy().astype(int)
    confs = boxes.conf.cpu().numpy()
    xyxy = boxes.xyxy.cpu().numpy()

    feat["yolo_num_objects"] = int(len(cls_ids))

    for i, cid in enumerate(cls_ids):
        x1, y1, x2, y2 = xyxy[i]
        box_area = (x2 - x1) * (y2 - y1) / img_area
        conf = float(confs[i])

        if cid == YOLO_CLASSES["person"]:
            if conf > feat["yolo_person_conf"]:
                feat["yolo_person_detected"] = 1
                feat["yolo_person_conf"] = conf
                feat["yolo_person_area"] = box_area

        elif cid == YOLO_CLASSES["phone"]:
            if conf > feat["yolo_phone_conf"]:
                feat["yolo_phone_conf"] = conf
                feat["yolo_phone_area"] = box_area
            phone_boxes.append((x1 / w, y1 / h, x2 / w, y2 / h, conf))

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

    return feat, phone_boxes


# ═══════════════════════════════════════════════════════════════════════════════
#  F. IMAGE-LEVEL (8 features — only invariant ones)
# ═══════════════════════════════════════════════════════════════════════════════

def compute_image_features(image):
    """8 structural image features that are less environment-dependent."""
    h, w = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    f = {}

    # Edge density (structural, not colour-dependent)
    edges = cv2.Canny(gray, 50, 150)
    f["img_edge_density"] = edges.mean() / 255.0
    cw1, cw2 = w // 4, 3 * w // 4
    ch1, ch2 = h // 4, 3 * h // 4
    f["img_edge_center"] = edges[ch1:ch2, cw1:cw2].mean() / 255.0

    # Texture variance (Laplacian — structural)
    f["img_texture_var"] = cv2.Laplacian(gray, cv2.CV_64F).var() / 10000.0

    # Skin ratio (useful but somewhat invariant)
    lower_skin = np.array([0, 20, 70])
    upper_skin = np.array([20, 255, 255])
    skin_mask = cv2.inRange(hsv, lower_skin, upper_skin)
    f["img_skin_ratio"] = skin_mask.mean() / 255.0
    f["img_skin_upper"] = skin_mask[:h//2, :].mean() / 255.0

    # Gradient magnitude (structural)
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    mag = np.sqrt(gx**2 + gy**2)
    f["img_gradient_mean"] = mag.mean() / 255.0

    # Phone-like bright rectangle detection (structural)
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
    f["img_phone_rect_count"] = rect_count

    return f


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN EXTRACT
# ═══════════════════════════════════════════════════════════════════════════════

def extract_features(image_path, label):
    """Extract ~78 generalised features from a single image."""
    image = cv2.imread(image_path)
    if image is None:
        return None

    h, w = image.shape[:2]
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    feat = {"label": label}

    # --- F. Image-level (8) ---
    feat.update(compute_image_features(image))

    # --- E. YOLO objects (14) + phone boxes ---
    yolo_feat, phone_boxes = compute_yolo_features(image)
    feat.update(yolo_feat)

    # --- MediaPipe detections ---
    face_results = face_mesh.process(rgb)
    hand_results = hands_det.process(rgb)
    pose_results = pose_det.process(rgb)

    face_detected = (face_results.multi_face_landmarks is not None and
                     len(face_results.multi_face_landmarks) > 0)
    feat["face_detected"] = int(face_detected)

    face_lm = face_results.multi_face_landmarks[0].landmark if face_detected else None
    face_w_val = 0.0
    face_h_val = 0.0

    # --- A. Face geometry (16) ---
    if face_detected:
        face_feat, face_w_val, face_h_val = compute_face_features(face_lm, h, w)
        feat.update(face_feat)
    else:
        face_keys = [
            "ear_left", "ear_right", "ear_avg", "mar",
            "head_pitch_deg", "head_yaw_deg", "head_roll_deg",
            "head_pitch_geom", "head_yaw_geom",
            "gaze_x", "gaze_y",
            "face_aspect_ratio", "face_area_rel",
            "eyebrow_left_dist_norm", "eyebrow_right_dist_norm",
        ]
        for k in face_keys:
            feat[k] = 0.0

    # --- B. Hand–face relations (14) ---
    feat.update(compute_hand_features(face_lm, hand_results, face_w_val, face_h_val))

    # --- C. Pose / body (16) ---
    feat.update(compute_pose_features(pose_results.pose_landmarks))

    # --- D. Phone-specific (10) ---
    feat.update(compute_phone_features(
        face_lm, pose_results.pose_landmarks, phone_boxes, face_w_val, face_h_val
    ))

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
    # Fill any NaN with 0 (landmarks not detected in some images)
    df = df.fillna(0.0)
    cols = ["label", "filename"] + [c for c in df.columns if c not in ("label", "filename")]
    df = df[cols]
    df.to_csv(OUTPUT_FILE, index=False)

    feat_cols = [c for c in df.columns if c not in ("label", "filename")]
    print(f"\nDone! Saved {len(df)} rows × {len(feat_cols)} features to {OUTPUT_FILE}")
    print(f"Skipped: {total_skipped}")
    print(f"\nFeatures ({len(feat_cols)}):")
    for c in feat_cols:
        print(f"  - {c}")


if __name__ == "__main__":
    main()
