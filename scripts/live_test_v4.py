"""
Live webcam behaviour detection — V4 (Rule-Based Priority Engine + YOLO + MediaPipe).

Priority cascade:
  1. NO USER      — No pose skeleton AND no YOLO person → skip everything else
  2. PHONE USING  — YOLO phone bbox overlaps wrist coords + nose angled down
  3. DISTRACTED   — Extreme head turn OR leaning back for 3+ consecutive seconds
  4. STUDYING     — Default: present, not on phone, not distracted, hands near desk/study objects

Press 'q' to quit, 'y' to toggle YOLO overlay, 'd' to toggle debug info.
"""

import os
import sys
import time
import cv2
import mediapipe as mp
import numpy as np
import warnings
from ultralytics import YOLO

warnings.filterwarnings("ignore")

# ─── YOLO setup ───────────────────────────────────────────────────────────────
print("Loading YOLOv8n model...")
yolo_model = YOLO("yolov8n.pt")

YOLO_CLASSES = {
    "person": 0, "phone": 67, "book": 73, "laptop": 63,
    "remote": 65, "mouse": 64, "keyboard": 66, "tv": 62,
    "bottle": 39, "cup": 41, "scissors": 76, "clock": 74,
    "backpack": 24, "handbag": 26,
}
YOLO_ID_TO_NAME = {v: k for k, v in YOLO_CLASSES.items()}
DISTRACTOR_IDS = {67, 65, 62, 74}
STUDY_OBJ_IDS  = {73, 63, 64, 66, 76}  # book, laptop, mouse, keyboard, scissors

YOLO_BOX_COLORS = {
    "distractor": (0, 0, 255),   # red
    "study":      (0, 200, 0),   # green
    "person":     (255, 200, 0), # cyan
    "other":      (200, 200, 0), # teal
}

# ─── MediaPipe init ───────────────────────────────────────────────────────────
mp_face_mesh = mp.solutions.face_mesh
mp_hands     = mp.solutions.hands
mp_pose      = mp.solutions.pose
mp_drawing   = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False, max_num_faces=1, refine_landmarks=True,
    min_detection_confidence=0.3, min_tracking_confidence=0.4,
)
hands_detector = mp_hands.Hands(
    static_image_mode=False, max_num_hands=2,
    min_detection_confidence=0.3, min_tracking_confidence=0.4,
)
pose_detector = mp_pose.Pose(
    static_image_mode=False, model_complexity=1,
    min_detection_confidence=0.3, min_tracking_confidence=0.4,
)

show_yolo  = True
show_debug = False

# ─── Tuneable thresholds ─────────────────────────────────────────────────────
# Phone-wrist proximity: how many pixels (normalised 0-1) wrist can be from
# phone bounding box edge to count as "holding phone"
PHONE_WRIST_MARGIN  = 0.06   # ~6% of frame dimension

# Head-turn: nose deviation from shoulder midpoint (normalised by shoulder width)
HEAD_TURN_THRESHOLD = 0.60   # >0.6 shoulder-widths off-centre = looking away

# Lean-back: if nose-to-avg-wrist distance > this fraction of frame height
LEAN_BACK_THRESHOLD = 0.45   # normalised

# How long distraction must persist before we label it (seconds)
DISTRACTION_TIME_S  = 3.0

# Nose downward angle threshold (pitch) for "looking down at phone"
NOSE_DOWN_PITCH     = -0.10  # negative pitch = looking down

# Gaze-towards-phone thresholds
GAZE_PHONE_MARGIN   = 0.18   # gaze vector must aim within this of phone centre
HEAD_YAW_PHONE_THR  = 0.25   # head yaw offset (normalised) threshold for looking at phone


# ═══════════════════════════════════════════════════════════════════════════════
#  YOLO helper
# ═══════════════════════════════════════════════════════════════════════════════

def run_yolo(image):
    """Return (detections_list, phone_boxes, study_boxes, person_detected).
    Each box is (x1,y1,x2,y2) in pixel coords."""
    h, w = image.shape[:2]
    detections = []
    phone_boxes = []
    study_boxes = []
    person_detected = False

    results = yolo_model(image, verbose=False, conf=0.25)
    if len(results) == 0 or results[0].boxes is None or len(results[0].boxes) == 0:
        return detections, phone_boxes, study_boxes, person_detected

    boxes   = results[0].boxes
    cls_ids = boxes.cls.cpu().numpy().astype(int)
    confs   = boxes.conf.cpu().numpy()
    xyxy    = boxes.xyxy.cpu().numpy()

    for i, cid in enumerate(cls_ids):
        x1, y1, x2, y2 = xyxy[i]
        conf = float(confs[i])
        name = YOLO_ID_TO_NAME.get(cid, None)
        if name is None:
            continue
        detections.append((int(x1), int(y1), int(x2), int(y2), cid, conf, name))

        if cid == YOLO_CLASSES["person"]:
            person_detected = True
        elif cid == YOLO_CLASSES["phone"]:
            phone_boxes.append((x1 / w, y1 / h, x2 / w, y2 / h))  # normalised
        if cid in STUDY_OBJ_IDS:
            study_boxes.append((x1 / w, y1 / h, x2 / w, y2 / h))

    return detections, phone_boxes, study_boxes, person_detected


# ═══════════════════════════════════════════════════════════════════════════════
#  MediaPipe helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _lm_xy(landmark):
    """Return (x, y) normalised coords from a single landmark."""
    return np.array([landmark.x, landmark.y])


def get_nose_pitch(face_landmarks):
    """Compute vertical pitch of the nose relative to the eye midpoint.
    Negative = looking down, positive = looking up."""
    lm = face_landmarks
    nose = np.array([lm[1].x, lm[1].y, lm[1].z])
    le   = np.array([lm[33].x, lm[33].y, lm[33].z])
    re   = np.array([lm[263].x, lm[263].y, lm[263].z])
    chin = np.array([lm[152].x, lm[152].y, lm[152].z])
    eye_mid = (le + re) / 2.0
    face_h  = np.linalg.norm(eye_mid[:2] - chin[:2])
    if face_h == 0:
        return 0.0
    pitch = (nose[1] - eye_mid[1]) / face_h
    return pitch


def get_head_yaw(pose_landmarks):
    """How far the nose is left/right of shoulder midpoint (normalised by shoulder width).
    Returns (offset_ratio, shoulder_mid_x, nose_x)."""
    plm = pose_landmarks
    ls = _lm_xy(plm[mp_pose.PoseLandmark.LEFT_SHOULDER.value])
    rs = _lm_xy(plm[mp_pose.PoseLandmark.RIGHT_SHOULDER.value])
    nose = _lm_xy(plm[mp_pose.PoseLandmark.NOSE.value])
    shoulder_mid_x = (ls[0] + rs[0]) / 2.0
    shoulder_width = abs(ls[0] - rs[0])
    if shoulder_width < 0.01:
        return 0.0, shoulder_mid_x, nose[0]
    offset = abs(nose[0] - shoulder_mid_x) / shoulder_width
    return offset, shoulder_mid_x, nose[0]


def get_lean_back_ratio(pose_landmarks):
    """Distance from nose to average wrist position, normalised.
    Large value = leaning far from hands / desk."""
    plm = pose_landmarks
    nose = _lm_xy(plm[mp_pose.PoseLandmark.NOSE.value])
    lw   = _lm_xy(plm[mp_pose.PoseLandmark.LEFT_WRIST.value])
    rw   = _lm_xy(plm[mp_pose.PoseLandmark.RIGHT_WRIST.value])
    avg_wrist = (lw + rw) / 2.0
    return np.linalg.norm(nose - avg_wrist)


def wrist_in_phone_box(pose_landmarks, phone_boxes, margin=PHONE_WRIST_MARGIN):
    """Check if either wrist falls inside (or near) any phone bounding box."""
    plm = pose_landmarks
    lw = _lm_xy(plm[mp_pose.PoseLandmark.LEFT_WRIST.value])
    rw = _lm_xy(plm[mp_pose.PoseLandmark.RIGHT_WRIST.value])
    for (bx1, by1, bx2, by2) in phone_boxes:
        for wx, wy in [lw, rw]:
            if (bx1 - margin <= wx <= bx2 + margin and
                by1 - margin <= wy <= by2 + margin):
                return True
    return False


def wrists_near_study_objects(pose_landmarks, study_boxes, margin=0.10):
    """Check if wrists overlap with any study-object bounding box."""
    plm = pose_landmarks
    lw = _lm_xy(plm[mp_pose.PoseLandmark.LEFT_WRIST.value])
    rw = _lm_xy(plm[mp_pose.PoseLandmark.RIGHT_WRIST.value])
    for (bx1, by1, bx2, by2) in study_boxes:
        for wx, wy in [lw, rw]:
            if (bx1 - margin <= wx <= bx2 + margin and
                by1 - margin <= wy <= by2 + margin):
                return True
    return False


def gaze_towards_phone(face_landmarks, phone_boxes, margin=GAZE_PHONE_MARGIN):
    """Check if the eye gaze vector points towards any phone bounding box.
    Uses iris landmarks (468, 473) relative to eye corners to estimate
    where the user is looking, then checks if that direction aligns with
    the phone centre."""
    lm = face_landmarks
    try:
        # Iris centres
        li = np.array([lm[468].x, lm[468].y])
        ri = np.array([lm[473].x, lm[473].y])
        # Eye corners for reference
        lc = (np.array([lm[133].x, lm[133].y]) + np.array([lm[33].x, lm[33].y])) / 2.0
        rc = (np.array([lm[362].x, lm[362].y]) + np.array([lm[263].x, lm[263].y])) / 2.0
        # Average gaze offset (normalised)
        gaze_offset = ((li - lc) + (ri - rc)) / 2.0
        # Estimated gaze landing point on frame (nose + gaze direction scaled)
        nose = np.array([lm[1].x, lm[1].y])
        gaze_point = nose + gaze_offset * 3.0  # scale to approximate landing

        for (bx1, by1, bx2, by2) in phone_boxes:
            pcx = (bx1 + bx2) / 2.0
            pcy = (by1 + by2) / 2.0
            # Check if gaze point lands near phone centre
            if abs(gaze_point[0] - pcx) < margin and abs(gaze_point[1] - pcy) < margin:
                return True
            # Also check if gaze point falls inside the phone bbox (expanded)
            if (bx1 - margin <= gaze_point[0] <= bx2 + margin and
                by1 - margin <= gaze_point[1] <= by2 + margin):
                return True
    except (IndexError, Exception):
        pass
    return False


def head_towards_phone(face_landmarks, phone_boxes, yaw_thr=HEAD_YAW_PHONE_THR):
    """Check if the head yaw/pitch orientation aligns with a detected phone.
    Returns True if the head is angled towards the phone's position."""
    lm = face_landmarks
    nose = np.array([lm[1].x, lm[1].y])
    le = np.array([lm[33].x, lm[33].y])
    re = np.array([lm[263].x, lm[263].y])
    eye_mid = (le + re) / 2.0
    ed = np.linalg.norm(le - re)
    if ed < 0.01:
        return False
    # Horizontal head direction (yaw); positive = looking right
    head_dir_x = (nose[0] - eye_mid[0]) / ed
    # Vertical head direction (pitch); positive = looking down
    chin = np.array([lm[152].x, lm[152].y])
    face_h = np.linalg.norm(eye_mid - chin)
    head_dir_y = (nose[1] - eye_mid[1]) / face_h if face_h > 0 else 0.0

    for (bx1, by1, bx2, by2) in phone_boxes:
        pcx = (bx1 + bx2) / 2.0
        pcy = (by1 + by2) / 2.0
        # Phone is to the right of nose → head should be angled right (positive yaw)
        phone_dx = pcx - nose[0]
        phone_dy = pcy - nose[1]
        # Check directional agreement: head yaw matches phone horizontal direction
        yaw_ok = (phone_dx * head_dir_x > 0) or abs(phone_dx) < 0.10
        # Phone is below face → head pitch should be downward (positive)
        pitch_ok = (phone_dy > 0 and head_dir_y > 0) or abs(phone_dy) < 0.08
        if yaw_ok and pitch_ok:
            return True
    return False


def wrists_on_desk(pose_landmarks, threshold=0.55):
    """Check if wrists are resting near the bottom half of frame (on desk)."""
    plm = pose_landmarks
    lw_y = plm[mp_pose.PoseLandmark.LEFT_WRIST.value].y
    rw_y = plm[mp_pose.PoseLandmark.RIGHT_WRIST.value].y
    return lw_y > threshold or rw_y > threshold


# ═══════════════════════════════════════════════════════════════════════════════
#  DRAW helpers
# ═══════════════════════════════════════════════════════════════════════════════

def draw_yolo_boxes(frame, detections):
    for (x1, y1, x2, y2, cid, conf, name) in detections:
        if cid == YOLO_CLASSES["person"]:
            color = YOLO_BOX_COLORS["person"]
        elif cid in DISTRACTOR_IDS:
            color = YOLO_BOX_COLORS["distractor"]
        elif cid in STUDY_OBJ_IDS:
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
    "no user":     (180, 180, 180),
    "phone using": (0, 0, 255),
    "distracted":  (0, 100, 255),
    "studying":    (0, 200, 0),
}

# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN LOOP
# ═══════════════════════════════════════════════════════════════════════════════

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("ERROR: Cannot open webcam")
    sys.exit(1)

current_label  = "Detecting..."
rule_reason    = ""
distraction_start = None          # timestamp when distraction was first detected
DISTRACTION_HOLD  = DISTRACTION_TIME_S

print("\n  Live test running — Rule-Based Priority Engine (V4)")
print("    Priority: No User > Phone Using > Distracted > Studying")
print("    q — Quit   y — Toggle YOLO   d — Toggle debug\n")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    fh, fw = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # ── Step A: Run YOLO ──────────────────────────────────────────────────
    yolo_dets, phone_boxes, study_boxes, yolo_person = run_yolo(frame)

    # ── Step B: Run MediaPipe Pose (primary skeleton check) ───────────────
    pose_results = pose_detector.process(rgb)
    pose_detected = pose_results.pose_landmarks is not None

    # ── Step C: Run MediaPipe Face Mesh (for nose pitch) ──────────────────
    face_results = face_mesh.process(rgb)
    face_detected = (face_results.multi_face_landmarks is not None
                     and len(face_results.multi_face_landmarks) > 0)

    # ══════════════════════════════════════════════════════════════════════
    #  RULE 1 — NO USER  (absolute priority)
    # ══════════════════════════════════════════════════════════════════════
    if not pose_detected and not yolo_person:
        current_label = "no user"
        rule_reason   = "No skeleton + no YOLO person"
        distraction_start = None

    else:
        # We have a user — proceed down the priority chain
        plm = pose_results.pose_landmarks.landmark if pose_detected else None

        # Get nose pitch from face mesh (more accurate than pose nose)
        nose_pitch = 0.0
        if face_detected:
            nose_pitch = get_nose_pitch(
                face_results.multi_face_landmarks[0].landmark
            )

        # ══════════════════════════════════════════════════════════════════
        #  RULE 2 — PHONE USING  (head + eyes towards phone)
        # ══════════════════════════════════════════════════════════════════
        phone_using = False
        phone_reason_parts = []
        if len(phone_boxes) > 0:
            # --- Check head orientation towards phone ---
            head_ok = False
            if face_detected:
                flm = face_results.multi_face_landmarks[0].landmark
                head_ok = head_towards_phone(flm, phone_boxes)
                if head_ok:
                    phone_reason_parts.append("head towards phone")

            # --- Check gaze / eyes towards phone ---
            gaze_ok = False
            if face_detected:
                flm = face_results.multi_face_landmarks[0].landmark
                gaze_ok = gaze_towards_phone(flm, phone_boxes)
                if gaze_ok:
                    phone_reason_parts.append("eyes towards phone")

            # --- Also check nose pitch (looking down at phone) ---
            nose_down = nose_pitch > NOSE_DOWN_PITCH
            if nose_down:
                phone_reason_parts.append("nose down")

            # --- Wrist near phone (holding it) ---
            wrist_overlap = False
            if plm is not None:
                wrist_overlap = wrist_in_phone_box(plm, phone_boxes)
                if wrist_overlap:
                    phone_reason_parts.append("wrist near phone")

            # Decision: phone detected AND (head+eyes towards it, OR
            #           head towards it + nose down, OR wrist holding + gaze/nose down)
            if head_ok and gaze_ok:
                phone_using = True
            elif head_ok and nose_down:
                phone_using = True
            elif gaze_ok and nose_down:
                phone_using = True
            elif wrist_overlap and (gaze_ok or nose_down or head_ok):
                phone_using = True
            elif wrist_overlap and nose_down:
                phone_using = True

        if phone_using:
            current_label = "phone using"
            rule_reason   = "YOLO phone + " + " + ".join(phone_reason_parts)
            distraction_start = None

        else:
            # ══════════════════════════════════════════════════════════════
            #  RULE 3 — DISTRACTED  (gaze & posture rule, 3-sec hold)
            # ══════════════════════════════════════════════════════════════
            is_distracted_now = False
            distract_detail   = ""

            if plm is not None:
                head_offset, smx, nx = get_head_yaw(plm)
                lean_ratio = get_lean_back_ratio(plm)

                if head_offset > HEAD_TURN_THRESHOLD:
                    is_distracted_now = True
                    distract_detail = f"Head turn {head_offset:.2f} > {HEAD_TURN_THRESHOLD}"

                if lean_ratio > LEAN_BACK_THRESHOLD:
                    is_distracted_now = True
                    distract_detail += (" + " if distract_detail else "") + \
                                       f"Lean back {lean_ratio:.2f} > {LEAN_BACK_THRESHOLD}"

            # Temporal gate: must persist for DISTRACTION_HOLD seconds
            if is_distracted_now:
                if distraction_start is None:
                    distraction_start = time.time()
                elapsed = time.time() - distraction_start
                if elapsed >= DISTRACTION_HOLD:
                    current_label = "distracted"
                    rule_reason   = distract_detail + f" ({elapsed:.1f}s)"
                else:
                    # Still counting — show studying with a warning
                    current_label = "studying"
                    rule_reason   = f"Studying (distraction pending {elapsed:.1f}/{DISTRACTION_HOLD}s)"
            else:
                distraction_start = None

                # ══════════════════════════════════════════════════════════
                #  RULE 4 — STUDYING  (default state)
                # ══════════════════════════════════════════════════════════
                study_detail = ""
                if plm is not None:
                    desk_ok    = wrists_on_desk(plm)
                    obj_ok     = wrists_near_study_objects(plm, study_boxes)
                    nose_center = True
                    head_off, _, _ = get_head_yaw(plm)
                    if head_off > HEAD_TURN_THRESHOLD:
                        nose_center = False

                    if obj_ok:
                        study_detail = "Hands near study objects"
                    elif desk_ok:
                        study_detail = "Hands on desk"
                    else:
                        study_detail = "Present, not distracted"
                    if nose_center:
                        study_detail += ", nose centred"

                current_label = "studying"
                rule_reason   = study_detail if study_detail else "User present (default)"

    # ── Build detected-objects string ─────────────────────────────────────
    obj_names = [d[6] for d in yolo_dets if d[4] != YOLO_CLASSES["person"]]
    detected_objects_str = ("Objects: " + ", ".join(sorted(set(obj_names)))) if obj_names else "Objects: none"

    # ── Draw overlays ─────────────────────────────────────────────────────
    if show_yolo:
        draw_yolo_boxes(frame, yolo_dets)

    # Face mesh
    if face_detected:
        for fl in face_results.multi_face_landmarks:
            mp_drawing.draw_landmarks(
                frame, fl, mp_face_mesh.FACEMESH_TESSELATION, None,
                mp_drawing_styles.get_default_face_mesh_tesselation_style()
            )

    # Pose skeleton
    if pose_detected:
        mp_drawing.draw_landmarks(
            frame, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
            mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=1),
        )

    # ── UI panel ──────────────────────────────────────────────────────────
    color = LABEL_COLORS.get(current_label, (255, 255, 255))

    panel_h = 165 if show_debug else 130
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (540, panel_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

    cv2.putText(frame, current_label.upper(), (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)

    cv2.putText(frame, f"Rule Engine V4  |  Priority: NoUser > Phone > Distracted > Studying",
                (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 180, 180), 1)

    cv2.putText(frame, detected_objects_str, (20, 105),
                cv2.FONT_HERSHEY_SIMPLEX, 0.50, (100, 220, 255), 1)

    if show_debug:
        cv2.putText(frame, f"Reason: {rule_reason}", (20, 130),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 255, 150), 1)

        debug_parts = []
        if pose_detected:
            plm = pose_results.pose_landmarks.landmark
            ho, _, _ = get_head_yaw(plm)
            lr = get_lean_back_ratio(plm)
            debug_parts.append(f"HeadOff={ho:.2f}")
            debug_parts.append(f"Lean={lr:.2f}")
        if face_detected:
            np_ = get_nose_pitch(face_results.multi_face_landmarks[0].landmark)
            debug_parts.append(f"Pitch={np_:.2f}")
        debug_parts.append(f"Phones={len(phone_boxes)}")
        debug_parts.append(f"StudyObj={len(study_boxes)}")
        cv2.putText(frame, "  ".join(debug_parts), (20, 155),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, (200, 200, 100), 1)

    cv2.putText(frame, "q: Quit | y: YOLO | d: Debug", (fw - 280, fh - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 150), 1)

    cv2.imshow("Behaviour Detection V4 — Rule Engine", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('y'):
        show_yolo = not show_yolo
        print(f"  YOLO overlay: {'ON' if show_yolo else 'OFF'}")
    elif key == ord('d'):
        show_debug = not show_debug
        print(f"  Debug info: {'ON' if show_debug else 'OFF'}")

cap.release()
cv2.destroyAllWindows()
print("Done.")
