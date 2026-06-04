import ctypes
import cv2
import pyautogui


def ctrl_space_pressed():
    ctrl = ctypes.windll.user32.GetAsyncKeyState(0x11)
    space = ctypes.windll.user32.GetAsyncKeyState(0x20)
    return ctrl and space


face_cascade = cv2.CascadeClassifier("face.xml")

camera = cv2.VideoCapture(0)

check_interval = 0.000000001
count = 0
faces = []
alt_tab_done = False
reset_key_pressed = False

while True:
    success, frame = camera.read()

    if not success:
        break

    count = count + 1

    if count >= check_interval:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.2,
            minNeighbors=5,
        )

        count = 0

    face_count = len(faces)

    for x, y, w, h in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.putText(
        frame,
        "Faces: " + str(face_count),
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2,
    )

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
