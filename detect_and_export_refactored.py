# detect_and_export.py
# 📌 리팩토링 목적:
# - input() 사용 제거 → argparse로 color 외부 인자 받기
# - grip_records.csv → data/{color}.json 자동 저장
# - 결과 이미지 → static/{color}.jpg로 저장

from ultralytics import YOLO
import cv2
import numpy as np
import mediapipe as mp
import csv
import os
import argparse
import json

# 📌 argparse로 외부에서 color 인자 받기
parser = argparse.ArgumentParser()
parser.add_argument('--color', type=str, required=True, help='선택할 홀드 색상 (예: red, blue 등)')
args = parser.parse_args()
selected_color = args.color.strip().lower()

# YOLO 모델 로드
model = YOLO("weights/best.pt")

# MediaPipe 초기화
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# 📌 영상 경로 고정 (기본 영상 하나만 사용 중이라면 추후 argparse 추가 가능)
video_path = "videos/0505_orange.mov.mp4"
cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
print(fps)
skip_frames = 0

ret, first_frame = cap.read()
if not ret:
    print("\u274c 첫 프레임을 읽을 수 없습니다.")
    cap.release()
    exit()

# 선택 가능한 색상 목록과 YOLO 클래스명 매핑
detected_classes = {
    'red': 'Hold_Red', 'orange': 'Hold_Orange', 'yellow': 'Hold_Yellow',
    'green': 'Hold_Green', 'blue': 'Hold_Blue', 'purple': 'Hold_Purple',
    'pink': 'Hold_Pink', 'white': 'Hold_White', 'black': 'Hold_Black',
    'gray': 'Hold_Gray', 'lime': 'Hold_Lime', 'sky': 'Hold_Sky',
}

if selected_color not in detected_classes:
    print("\u274c 유효하지 않은 색상입니다. 종료합니다.")
    cap.release()
    exit()

selected_class = detected_classes[selected_color]
print(f"🎯 선택된 클래스: {selected_class}")

first_frame = cv2.rotate(first_frame, cv2.ROTATE_90_CLOCKWISE)
result = model(first_frame)[0]

if result.masks is None:
    print("\u274c 마스크 없음 (YOLO 결과)")
    cap.release()
    exit()

# 색상에 따른 BGR 색상값 정의
color_map = {
    'Hold_Red': (0, 0, 255), 'Hold_Orange': (0, 165, 255), 'Hold_Yellow': (0, 255, 255),
    'Hold_Green': (0, 255, 0), 'Hold_Blue': (255, 0, 0), 'Hold_Purple': (204, 50, 153),
    'Hold_Pink': (203, 192, 255), 'Hold_Lime': (50, 255, 128), 'Hold_Sky': (255, 255, 0),
    'Hold_White': (255, 255, 255), 'Hold_Black': (30, 30, 30), 'Hold_Gray': (150, 150, 150),
}

masks = result.masks.data
boxes = result.boxes
names = model.names
img_h, img_w = first_frame.shape[:2]
hold_contours = []

# YOLO 마스크로부터 contour 추출
for i in range(masks.shape[0]):
    mask = masks[i].cpu().numpy()
    resized_mask = cv2.resize(mask, (img_w, img_h), interpolation=cv2.INTER_NEAREST)
    binary_mask = (resized_mask > 0.7).astype(np.uint8) * 255
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    cls_id = int(boxes.cls[i].item())
    conf = float(boxes.conf[i].item())
    class_name = names[cls_id]
    color = color_map.get(class_name, (255, 255, 255))

    if class_name == selected_class and contours:
        contour = contours[0]
        M = cv2.moments(contour)
        cx, cy = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])) if M["m00"] != 0 else (0, 0)
        hold_contours.append({
            "class_name": class_name,
            "contour": contours[0],
            "color": color,
            "label": f"{class_name} {conf:.2f}",
            "center": (cx, cy)
        })

# CSV 파일 작성 준비
os.makedirs("data", exist_ok=True)
csv_path = f'data/hand_foot_coordinates.csv'
csv_file = open(csv_path, 'w', newline='')
csv_writer = csv.writer(csv_file)
csv_writer.writerow(['frame_idx','left_hand_x','left_hand_y','right_hand_x','right_hand_y'])

frame_idx = 0
out = None
os.makedirs("outputs", exist_ok=True)
os.makedirs("static", exist_ok=True)
grip_records = []
already_grabbed = {}

with mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, model_complexity=1) as pose:
    touch_counters = {i: 0 for i in range(len(hold_contours))}
    TOUCH_THRESHOLD = 10

    first_vis_frame_saved = False
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        for _ in range(skip_frames):
            cap.read()

        frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

        if out is None:
            height, width = frame.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_output_path = f'static/{selected_color}_climb.mp4'
            out = cv2.VideoWriter(video_output_path, fourcc, fps // (skip_frames + 1), (width, height))


        vis_frame = frame.copy()

        def invert_color(bgr):
            return (255 - bgr[0], 255 - bgr[1], 255 - bgr[2])

        for hold in hold_contours:
            cv2.drawContours(vis_frame, [hold["contour"]], -1, hold["color"], 2)
            x, y, w, h = cv2.boundingRect(hold["contour"])
            cv2.putText(vis_frame, hold["label"], (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, hold["color"], 2)
            cx, cy = hold["center"]
            inverted = invert_color(hold["color"])
            cv2.circle(vis_frame, (cx, cy), 4, inverted, -1)
            cv2.putText(vis_frame, f"({cx}, {cy})", (cx + 6, cy - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.4, inverted, 1)
        
        # while 루프 내부에서 vis_frame 시각화 직후
        if not first_vis_frame_saved:
            cv2.imwrite(f'static/{selected_color}.jpg', vis_frame)
            first_vis_frame_saved = True

        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = pose.process(image_rgb)

        important_landmarks = {"left_index": 15, "right_index": 16}
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
                    joint_color = (0, 0, 255)
                    cv2.circle(vis_frame, (int(x), int(y)), 5, joint_color, -1)
                    cv2.putText(vis_frame, f"{name}: ({int(x)}, {int(y)})", (int(x)+5, int(y)-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)

            current_touched = set()
            for name, (x, y) in coords.items():
                for i, hold in enumerate(hold_contours):
                    if cv2.pointPolygonTest(hold["contour"], (x, y), False) >= 0:
                        current_touched.add(i)

            for name, (x, y) in coords.items():
                for i, hold in enumerate(hold_contours):
                    if cv2.pointPolygonTest(hold["contour"], (x, y), False) >= 0:
                        key = (name, i)
                        touch_counters[key] = touch_counters.get(key, 0) + 1

                        if touch_counters[key] >= TOUCH_THRESHOLD:
                            cv2.drawContours(vis_frame, [hold["contour"]], -1, hold["color"], thickness=cv2.FILLED)
                        if already_grabbed.get(key) is None:
                            cx, cy = hold["center"]
                            grip_records.append([name, i, cx, cy])
                            already_grabbed[key] = True
                    else:
                        touch_counters[(name, i)] = 0

        cv2.putText(vis_frame, f"Frame {frame_idx}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
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

# 📌 grip_records를 JSON으로 저장
json_path = f'data/{selected_color}.json'
with open(json_path, "w") as f:
    json.dump([
        {"part": part, "hold_id": hold_id, "cx": cx, "cy": cy}
        for part, hold_id, cx, cy in grip_records
    ], f, indent=2)

print(f"\n✅ {selected_color} 경로 분석 완료: 좌표 저장 → {json_path}, 이미지 저장 → static/{selected_color}.jpg")
