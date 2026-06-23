import ctypes
import cv2
import pyautogui
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision


def ctrl_space_pressed():
    ctrl = ctypes.windll.user32.GetAsyncKeyState(0x11)
    space = ctypes.windll.user32.GetAsyncKeyState(0x20)
    return ctrl and space


def merge_boxes(boxes, iou_threshold=0.3):
    merged = []
    for box in boxes:
        x, y, w, h = box
        duplicate = False
        for mx, my, mw, mh in merged:
            ix1 = max(x, mx)
            iy1 = max(y, my)
            ix2 = min(x + w, mx + mw)
            iy2 = min(y + h, my + mh)   
            iw = max(0, ix2 - ix1)
            ih = max(0, iy2 - iy1)
            inter = iw * ih
            union = w * h + mw * mh - inter
            if union > 0 and inter / union > iou_threshold:
                duplicate = True
                break
        if not duplicate:
            merged.append((x, y, w, h))
    return merged


def count_fingers(lm):
    count = 0

    margin = 0.03

    for tip in (8, 12, 16, 20):
        if lm[tip].y < lm[tip - 2].y - margin:
            count += 1

    thumb_dir = lm[2].x - lm[17].x
    if (lm[4].x - lm[3].x) * thumb_dir > 0:
        count += 1

    return count


face_cascade = cv2.CascadeClassifier("face.xml")

yunet = cv2.FaceDetectorYN.create(
    "face_detection_yunet_2023mar.onnx",
    "",
    (320, 320),
    score_threshold=0.6,
)

with open("hand_landmarker.task", "rb") as _f:
    _hand_model = _f.read()

hand_landmarker = vision.HandLandmarker.create_from_options(
    vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_buffer=_hand_model),
        num_hands=1,
        min_hand_detection_confidence=0.6,
        min_tracking_confidence=0.6,
    )
)

camera = cv2.VideoCapture(0)

faces = []
alt_tab_done = False
reset_key_pressed = False

hand_return_done = False
open_hand_frames = 0
OPEN_HAND_THRESHOLD = 5

while True:
    success, frame = camera.read()

    if not success:
        break

    h, w = frame.shape[:2]
    boxes = []

    yunet.setInputSize((w, h))
    _, yn_faces = yunet.detect(frame)
    if yn_faces is not None:
        for f in yn_faces:
            bx, by, bw, bh = f[:4].astype(int)
            boxes.append((bx, by, bw, bh))

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    haar_faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=4,
        minSize=(60, 60),
    )
    for x, y, fw, fh in haar_faces:
        boxes.append((int(x), int(y), int(fw), int(fh)))

    faces = merge_boxes(boxes)

    face_count = len(faces)

    for x, y, fw, fh in faces:
        cv2.rectangle(frame, (x, y), (x + fw, y + fh), (0, 255, 0), 2)

    cv2.putText(
        frame,
        "Faces: " + str(face_count),
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2,
    )

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = hand_landmarker.detect(mp_image)
    fingers = 0
    if result.hand_landmarks:
        fingers = count_fingers(result.hand_landmarks[0])

    cv2.putText(
        frame,
        "Fingers: " + str(fingers),
        (20, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 0, 0),
        2,
    )

    if fingers == 5:
        open_hand_frames += 1
    else:
        open_hand_frames = 0
        hand_return_done = False

    if open_hand_frames >= OPEN_HAND_THRESHOLD and not hand_return_done and alt_tab_done:
        pyautogui.hotkey("alt", "tab")
        hand_return_done = True

    if ctrl_space_pressed() and reset_key_pressed == False:
        alt_tab_done = False
        reset_key_pressed = True

    if not ctrl_space_pressed():
        reset_key_pressed = False

    if face_count >= 2 and alt_tab_done == False:
        pyautogui.hotkey("alt", "tab")
        alt_tab_done = True

    cv2.imshow("Face Detection", frame)

    if cv2.waitKey(1) == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()
