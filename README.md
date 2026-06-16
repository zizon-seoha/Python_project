# 얼굴 인식 + 손동작 창 전환

웹캠으로 **얼굴 수**와 **손가락 수**를 실시간 인식해서 Windows 창을 자동 전환하는 프로그램.

- **얼굴 2개 이상 감지** → 다른 창으로 전환 (`Alt+Tab`)
- **손가락 5개(하이파이브)** → 직전 창으로 복귀 (`Alt+Tab`) — *단, 얼굴로 한번 전환된 뒤에만 동작*

---

## 동작 흐름

```
카메라 프레임
   │
   ├─ 얼굴 검출 (YuNet + Haar) ── 박스 병합 ── 얼굴 수 카운트
   │        └─ 2개 이상 → Alt+Tab (다른 창으로)  [alt_tab_done = True]
   │
   └─ 손 검출 (MediaPipe) ── 손가락 수 카운트
            └─ 5개를 N프레임 연속 + alt_tab_done → Alt+Tab (직전 창 복귀)
```

핵심 상태값:

| 변수 | 역할 |
|------|------|
| `alt_tab_done` | 얼굴로 화면 전환이 일어났는지. 손동작 복귀의 **전제 조건** |
| `open_hand_frames` | 손가락 5개가 연속으로 잡힌 프레임 수 (오작동 방지) |
| `hand_return_done` | 손동작 복귀가 이미 실행됐는지 (한 번만 동작) |

---

## 필요 파일

| 파일 | 설명 |
|------|------|
| `main.py` | 본체 |
| `face.xml` | Haar Cascade 얼굴 검출기 |
| `face_detection_yunet_2023mar.onnx` | YuNet 딥러닝 얼굴 검출 모델 |
| `hand_landmarker.task` | MediaPipe 손 관절 검출 모델 (약 7.8MB) |

> `hand_landmarker.task`는 [MediaPipe 모델 페이지](https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task)에서 받아 프로젝트 폴더에 둔다.

## 설치 / 실행

```bash
pip install opencv-contrib-python mediapipe==0.10.14 pyautogui
python main.py
```

- 종료: 영상 창에서 **`q`** 키
- 리셋: **`Ctrl + Space`** → `alt_tab_done`을 풀어 얼굴 전환을 다시 가능하게 함

> **주의 — MediaPipe 버전**: `mediapipe==0.10.14`로 고정. 최신(0.10.35)은 손가락 인식에 쓰는 `solutions` API가 빠져 있고, 또 모델을 **bytes로 읽어 넘긴다**(한글 폴더 경로 `얼굴인식`을 C++ 로더가 못 읽는 문제 회피).

---

## 코드 해석

### 1. 키 입력 감지 — `ctrl_space_pressed()`

```python
def ctrl_space_pressed():
    ctrl = ctypes.windll.user32.GetAsyncKeyState(0x11)   # 0x11 = Ctrl
    space = ctypes.windll.user32.GetAsyncKeyState(0x20)  # 0x20 = Space
    return ctrl and space
```

Windows API(`GetAsyncKeyState`)로 Ctrl과 Space가 동시에 눌렸는지 본다. 리셋 트리거로 사용.

### 2. 겹친 얼굴 박스 합치기 — `merge_boxes()`

YuNet과 Haar 두 검출기 결과를 합치면 같은 얼굴이 2개로 잡힐 수 있다. **IoU(겹친 비율)** 가 `iou_threshold`(0.3)를 넘으면 같은 얼굴로 보고 하나만 남겨 중복 카운트를 막는다.

```python
inter = iw * ih                       # 교집합 넓이
union = w*h + mw*mh - inter           # 합집합 넓이
if inter / union > iou_threshold:     # 많이 겹치면 중복
    duplicate = True
```

### 3. 손가락 개수 세기 — `count_fingers(lm)`

MediaPipe가 주는 손 관절 21개 좌표(`lm`)로 펴진 손가락을 센다.

**검지~새끼 (4개)** — 손끝(tip)이 두 번째 관절(pip)보다 위(y가 작음)에 있으면 펴짐:
```python
margin = 0.03   # 경계에서 깜빡임 방지: 이만큼 확실히 위일 때만 인정
for tip in (8, 12, 16, 20):
    if lm[tip].y < lm[tip - 2].y - margin:
        count += 1
```

**엄지 (1개)** — 방향 기반 판정 (좌우손·카메라 미러 무관, 임계값 튜닝 불필요):
```python
thumb_dir = lm[2].x - lm[17].x              # 엄지 바깥 방향 (새끼뿌리→엄지뿌리)
if (lm[4].x - lm[3].x) * thumb_dir > 0:     # 끝(4)이 그 바깥으로 더 나가면 펴짐
    count += 1
```
엄지를 손바닥 안쪽으로 접으면 `tip`이 반대 방향으로 들어가 곱이 음수가 되어 세지 않는다.

> MediaPipe 손 관절 번호: 4=엄지끝, 8=검지끝, 12=중지끝, 16=약지끝, 20=새끼끝. 각 손끝에서 −2 하면 그 손가락의 pip 관절.

### 4. 얼굴 검출 (메인 루프)

두 방식을 같이 써서 정확도를 높인다.

- **YuNet** (`FaceDetectorYN`): 딥러닝 기반, 주 검출기. `setInputSize`로 프레임 크기를 매번 맞춰준다.
- **Haar Cascade**: 보조. 흑백 변환 + `equalizeHist`(명암 평탄화) 후 검출.

두 결과를 `merge_boxes`로 합쳐 최종 얼굴 수(`face_count`)를 구한다.

### 5. 손 검출 (메인 루프)

```python
rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)              # OpenCV는 BGR, MediaPipe는 RGB
mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
result = hand_landmarker.detect(mp_image)
if result.hand_landmarks:
    fingers = count_fingers(result.hand_landmarks[0])
```

### 6. 트리거 로직

**얼굴 → 다른 창:**
```python
if face_count >= 2 and alt_tab_done == False:
    pyautogui.hotkey("alt", "tab")
    alt_tab_done = True          # 한 번만. 다시 하려면 Ctrl+Space로 리셋
```

**손가락 5개 → 직전 창 복귀:**
```python
if fingers == 5:
    open_hand_frames += 1        # 연속 카운트
else:
    open_hand_frames = 0
    hand_return_done = False     # 손 내리면 리셋 → 다시 펴면 또 동작

if open_hand_frames >= OPEN_HAND_THRESHOLD and not hand_return_done and alt_tab_done:
    pyautogui.hotkey("alt", "tab")   # 직전 창으로 토글백
    hand_return_done = True
```
- `OPEN_HAND_THRESHOLD`(=5)프레임 **연속**으로 5개일 때만 → 순간 오인식 무시
- `alt_tab_done` 조건 → 얼굴로 화면이 바뀐 뒤에만 복귀 동작

---

## 튜닝 포인트

| 증상 | 조정 |
|------|------|
| 손가락 깜빡임/오인식 | `count_fingers`의 `margin` ↑ (0.04~0.05) |
| 엄지가 거꾸로 (폈는데 안 셈 / 접었는데 셈) | 엄지 판정 부등호 `> 0` ↔ `< 0` |
| 손동작이 너무 민감/둔함 | `OPEN_HAND_THRESHOLD` 조정 |
| 얼굴이 과검출 | YuNet `score_threshold`, Haar `minNeighbors`/`minSize` 조정 |
