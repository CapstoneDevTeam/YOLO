from ultralytics import YOLO
import cv2
import numpy as np
import mediapipe as mp
import csv
import os

# YOLO 모델 로드
model = YOLO("weights/best.pt")

# MediaPipe 초기화
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# 영상 열기
video_path = "videos/0505_orange.mov.mp4"
cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
print(fps)
skip_frames = 0

# cv2.VideoCapture.read함수는 튜플 형태로 값을 반환
# ret는 Ture / Flase
# first_frame은 첫 프레임. 만약 read를 한번 더 한다면 그 다음 프레임을 가져옴
# 이때 이미지를 가져올 때 Numpy 배열 형태로 가져옴
ret, first_frame = cap.read()
if not ret:
    print("❌ 첫 프레임을 읽을 수 없습니다.")
    cap.release()
    exit()

# dictionary 구조. key와 value 구조.
# value값을 YOLO class 이름과 맞춤춤
all_colors = {
    'red': 'Hold_Red',
    'orange': 'Hold_Orange',
    'yellow': 'Hold_Yellow',
    'green': 'Hold_Green',
    'blue': 'Hold_Blue',
    'purple': 'Hold_Purple',
    'pink': 'Hold_Pink',
    'white': 'Hold_White',
    'black': 'Hold_Black',
    'gray': 'Hold_Gray',
    'lime': 'Hold_Lime',
    'sky': 'Hold_Sky',
}
#join을 통해 문자열 쫙 보여줌.
print("🎨 선택 가능한 색상: " + ", ".join(all_colors.keys()))
#strip은 공백 제거, lower은 다 소문자로 변환
selected_color = input("✅ 원하는 홀드 색상 입력: ").strip().lower()

if selected_color not in all_colors:
    print("❌ 유효하지 않은 색상입니다. 종료합니다.")
    exit()

selected_class = all_colors[selected_color]
print(f"🎯 선택된 클래스: {selected_class}")

#화면 회전 없어도 되는게 있음
first_frame = cv2.rotate(first_frame, cv2.ROTATE_90_CLOCKWISE)
#openCV가 BGR을 써서 그냥 BGR을 이용해버리자 해서 rgb로 바꾸지 않았음.
#first_rgb = cv2.cvtColor(first_frame, cv2.COLOR_BGR2RGB)
# model은 YOLO지. 근데 YOLO가 값을 반환할 때 리스트로 반환함. 그 첫번째 리스트의 정보를 받기 위해[0]을 붙임.
result = model(first_frame)[0]

if result.masks is None:
    print("❌ 마스크 없음 (YOLO 결과)")
    cap.release()
    exit()

print("🎯 YOLO 탐지 개수:", result.masks.shape[0])


#BGR기준으로 작성함.
color_map = {
    'Hold_Red':     (0, 0, 255),       # 🔴
    'Hold_Orange':  (0, 165, 255),     # 🟠
    'Hold_Yellow':  (0, 255, 255),     # 🟡
    'Hold_Green':   (0, 255, 0),       # 🟢
    'Hold_Blue':    (255, 0, 0),       # 🔵
    'Hold_Purple':  (204, 50, 153),    # 🟣
    'Hold_Pink':    (203, 192, 255),   # 💗
    'Hold_Lime':    (50, 255, 128),    # 연두
    'Hold_Sky':     (255, 255, 0),     # 하늘색
    'Hold_White':   (255, 255, 255),   # ⚪
    'Hold_Black':   (30, 30, 30),      # ⚫
    'Hold_Gray':    (150, 150, 150),   # 회색
}

# YOLO 마스크 → contour 추출
# masks.data를 해야 픽셀의 3차원 배열 나옴
# YOLO가 감지한 객체 수, 마스크 이미지의 높이, 마스크 이미지의 넓이가 담겨있음.
# 그래서 masks의 3차원 배열을 출력해본다면 0~1의 값이 담겨있는데 그게 객체일 확률인듯
masks = result.masks.data
# 이건 바운딩박스 제공인데 필요 없긴 해
boxes = result.boxes
# model.names는 YOLO가 인식할 수 있도록 학습된 클래스 ID ↔ 클래스 이름 매핑 딕셔너리
# 왜 필요하냐면 YOLO는 class이름에는 관심이 없고 class ID만 봄. 그래서 우리가 알아볼 수 있게 class 이름으로 교체.
names = model.names
# [:2]는 처음부터 2번째까지 값만 쓰자.그게 이미지의 높이와 넓이임.
img_h, img_w = first_frame.shape[:2]
# 빈 리스트 선언, 윤곽선에 대한 정보 담을거임.
hold_contours = []  # 얘는 mediapipe 영상 처리에만 사용
# yolo 돌려서 홀드 좌표들은 한 번에 색상별로 저장하는게 좋을 거 같다고 생각함
hold_contours_all = {color: [] for color in all_colors}  # 얘는 YOLO 결과 전부 저장해버리기

# masks.shape[0]은 객체 검출한 수, 즉 홀드 수 만큼 반복함
for i in range(masks.shape[0]):
    # GPU tensor → numpy 변환
    mask = masks[i].cpu().numpy()

    # YOLO가 리사이즈한 마스크 → 원래 해상도로 복원
    resized_mask = cv2.resize(mask, (img_w, img_h), interpolation=cv2.INTER_NEAREST)
    binary_mask = (resized_mask > 0.7).astype(np.uint8) * 255

    # 윤곽선 추출
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        continue

    # 클래스 정보 추출
    cls_id = int(boxes.cls[i].item())
    conf = float(boxes.conf[i].item())
    class_name = names[cls_id]
    color = color_map.get(class_name, (255, 255, 255))  # 시각화용 BGR 색상

    # YOLO class_name → 우리가 정한 color_key (예: Hold_Red → red)
    color_key = next((k for k, v in all_colors.items() if v == class_name), None)
    if color_key is None:
        continue

    # 중심좌표 계산
    contour = contours[0]
    M = cv2.moments(contour)
    cx, cy = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])) if M["m00"] != 0 else (0, 0)

    # 바운더리 박스 계산 (중심 기준 상하좌우 최대거리)
    mask_points = contour[:, 0, :]  # shape: (N, 2)
    x_left   = min(mask_points[mask_points[:, 0] < cx][:, 0], default=cx)
    x_right  = max(mask_points[mask_points[:, 0] > cx][:, 0], default=cx)
    y_top    = min(mask_points[mask_points[:, 1] < cy][:, 1], default=cy)
    y_bottom = max(mask_points[mask_points[:, 1] > cy][:, 1], default=cy)

    # 홀드 정보 저장
    hold_info = {
        "class_name": class_name,
        "contour": contour,
        "color": color,
        "label": f"{class_name} {conf:.2f}",
        "center": (cx, cy),
        "boundary": (x_left, y_top, x_right, y_bottom),
        "index": len(hold_contours_all[color_key])
    }
    hold_contours_all[color_key].append(hold_info)
# 선택한 색상에 대한 홀드 정보만 추출 → mediapipe 처리용
hold_contours = hold_contours_all[selected_color]

# CSV 파일 준비
csv_file = open('data/hand_foot_coordinates.csv', 'w', newline='')
csv_writer = csv.writer(csv_file)
csv_writer.writerow([
    'frame_idx',
    'left_hand_x', 'left_hand_y',
    'right_hand_x', 'right_hand_y',
])

# 프레임 인덱스
frame_idx = 0
# 출력 영상 만들려고 설정정
out = None

grip_records = []  # [part, hold_id, cx, cy]
# 이미 잡은 홀드인지 판단, 중복 방지용.
already_grabbed = {}

# Pose 추론 루프
# with 구문은 파이썬의 context manager 문법으로, 자원(메모리 등)을 자동으로 열고 닫아주는 역할.
# mp_pose.Pose를 통해 Midiapipe의 Pose모듈 불러옴.
# 비디오 스트림(=연속 프레임) 을 처리할 때는 False로 해야 빠름.
# 사람을 검출할 최소 신뢰도. 0.5보다 낮으면 추론 안 함
# 모델의 복잡도 (0: 빠름, 1: 중간, 2: 정확하지만 느림) – 보통 1이면 충분
# as pose로 pose라는 이름 붙임임
with mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, model_complexity=1) as pose:
    # 각 홀드별 터치 프레임 카운터 초기화. 다 0으로 맞춰줌.
    touch_counters = {i: 0 for i in range(len(hold_contours))}
    TOUCH_THRESHOLD = 10  # 최소 10프레임 연속 접촉해야 진짜 터치 -> 0.33333초(30프레임 영상이니니)

    #위에서 한거랑 똑같음. 영상 여는거.
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        #프레임 스킵해주는 건데 상윤쓰가 최적화 한다고 넣었음. 근데 Mediapipe를 최적화 해야되는거 같음. 프레임을 넘기는게 아니라.
        for _ in range(skip_frames):
            cap.read()

        #화면 회전 없어도 되는게 있음
        frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        
        # 영상 저장하기 위해서 초기 설정정
        if out is None:
            height, width = frame.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter('outputs/ex4.mp4', fourcc, fps // (skip_frames + 1), (width, height))

        # 시각화 할 프레임 복사본. 원본에 하면 원본 훼손될거잖아.
        # .copy()는 Numpy에서 제공하는 Method
        vis_frame = frame.copy()

        def invert_color(bgr):
            return (255 - bgr[0], 255 - bgr[1], 255 - bgr[2])

        for hold in hold_contours:
            cv2.drawContours(vis_frame, [hold["contour"]], -1, hold["color"], 2)
            x, y, w, h = cv2.boundingRect(hold["contour"])
            cv2.putText(vis_frame, hold["label"], (x, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, hold["color"], 2, cv2.LINE_AA)
            
            #중심점 그리기
            cx, cy = hold["center"]
            inverted = invert_color(hold["color"])
            cv2.circle(vis_frame, (cx, cy), 4, inverted, -1)
            cv2.putText(vis_frame, f"({cx}, {cy})", (cx + 6, cy - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, inverted, 1, cv2.LINE_AA)

        # Mediapipe pose 추론
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = pose.process(image_rgb)

        important_landmarks = {
            "left_index": 15,
            "right_index": 16,
        }

        hand_parts = {"left_index", "right_index"}

        if result.pose_landmarks:
            h, w = frame.shape[:2]
            coords = {}

            for name, idx in important_landmarks.items():
                landmark = result.pose_landmarks.landmark[idx]
                coords[name] = (landmark.x * w, landmark.y * h)

            csv_writer.writerow([
                frame_idx,
                coords["left_index"][0], coords["left_index"][1],
                coords["right_index"][0], coords["right_index"][1],
            ])
            
            for name, (x, y) in coords.items():
                if name in hand_parts:
                    joint_color = (0, 0, 255)  # 손: 빨간색
                cv2.circle(vis_frame, (int(x), int(y)), 5, joint_color, -1)  # 원 그리기
                cv2.putText(vis_frame, f"{name}: ({int(x)}, {int(y)})", (int(x)+5, int(y)-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1, cv2.LINE_AA)

            
            # 현재 프레임에서 접촉된 홀드 인덱스 저장
            current_touched = set()

            for name, (x, y) in coords.items():
                for i, hold in enumerate(hold_contours):
                    if cv2.pointPolygonTest(hold["contour"], (x, y), False) >= 0:
                        current_touched.add(i)

            # 터치 카운트 갱신 + 색칠 여부 판단
            for name, (x, y) in coords.items():
                for i, hold in enumerate(hold_contours):
                    if cv2.pointPolygonTest(hold["contour"], (x, y), False) >= 0:
                        key = (name, i)
                        touch_counters[key] = touch_counters.get(key, 0) + 1

                        if touch_counters[key] >= TOUCH_THRESHOLD:
                            # 색칠
                            cv2.drawContours(vis_frame, [hold["contour"]], -1, hold["color"], thickness=cv2.FILLED)

                            # grip 기록
                        if already_grabbed.get(key) is None:
                            cx, cy = hold["center"]
                            grip_records.append([name, i, cx, cy])
                            already_grabbed[key] = True
                    else:
                        touch_counters[(name, i)] = 0



        cv2.putText(vis_frame, f"Frame {frame_idx}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

        # ▶️ 최종 영상 리사이즈 후 출력
        vh, vw = vis_frame.shape[:2]
        scale = 400 / vw
        resized = cv2.resize(vis_frame, (int(vw * scale), int(vh * scale)))

        cv2.imshow("YOLO + Pose Landmarks", resized)
        out.write(vis_frame)
        frame_idx += 1

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
csv_file.close()
out.release()
cv2.destroyAllWindows()

# 모든 색상 홀드 정보를 하나의 CSV로 저장
os.makedirs("data", exist_ok=True)
with open("data/all_bounding_boxes.csv", "w", newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["color", "index", "cx", "cy", "x_left", "y_top", "x_right", "y_bottom"])
    
    for color, holds in hold_contours_all.items():
        for hold in holds:
            cx, cy = hold["center"]
            x_left, y_top, x_right, y_bottom = hold["boundary"]
            writer.writerow([color, hold["index"], cx, cy, x_left, y_top, x_right, y_bottom])


with open("data/grip_records.csv", "w", newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["part", "hold_id", "cx", "cy"])
    writer.writerows(grip_records)

# 출력 영상
# out = cv2.VideoWriter('outputs/ex4.mp4', ...)

# 여기는 bounding_boxes에 해당하는 좌표값들 제대로 얻어진건지 파악하기 위해 좌표값들 연결해서 윤곽선 그려본 것. 제대로 나옴
# YOLO 인식만 홀드에 정확하게 된다면 코드 그대로 써도 될 듯
# ✅ 모든 색상에 대한 디버그 이미지 생성 및 저장
for color, holds in hold_contours_all.items():
    debug_img = first_frame.copy()
    for hold in holds:
        index = hold["index"]
        cx, cy = hold["center"]
        x_left, y_top, x_right, y_bottom = hold["boundary"]

        # 사각형 그리기
        cv2.rectangle(debug_img, (x_left, y_top), (x_right, y_bottom), (0, 255, 255), 2)
        # 중심 좌표 표시
        cv2.circle(debug_img, (cx, cy), 4, (0, 0, 255), -1)
        # 인덱스 + 색상명 표시
        cv2.putText(debug_img, f"{color}_{index}", (x_left, y_top - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

    # 이미지 저장 (예: data/debug_red.jpg)
    cv2.imwrite(f"data/debug_{color}.jpg", debug_img)

print("✅ 모든 색상 디버그 이미지 저장 완료!")


