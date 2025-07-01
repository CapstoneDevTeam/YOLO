# 색상별 홀드 이미지 표현할 때는 굳이 동영상 다 돌릴 필요 없음
# extract_holds_by_color.py
from ultralytics import YOLO
import cv2
import numpy as np
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('--color', type=str, required=True)
args = parser.parse_args()
selected_color = args.color.strip().lower()

color_class_map = {
    'red': 'Hold_Red', 'orange': 'Hold_Orange', 'yellow': 'Hold_Yellow',
    'green': 'Hold_Green', 'blue': 'Hold_Blue', 'purple': 'Hold_Purple',
    'pink': 'Hold_Pink', 'white': 'Hold_White', 'black': 'Hold_Black',
    'gray': 'Hold_Gray', 'lime': 'Hold_Lime', 'sky': 'Hold_Sky',
}
selected_class = color_class_map.get(selected_color)
if not selected_class:
    print("❌ 유효하지 않은 색상")
    exit()

color_map = {
    'Hold_Red': (0, 0, 255), 'Hold_Orange': (0, 165, 255), 'Hold_Yellow': (0, 255, 255),
    'Hold_Green': (0, 255, 0), 'Hold_Blue': (255, 0, 0), 'Hold_Purple': (204, 50, 153),
    'Hold_Pink': (203, 192, 255), 'Hold_Lime': (50, 255, 128), 'Hold_Sky': (255, 255, 0),
    'Hold_White': (255, 255, 255), 'Hold_Black': (30, 30, 30), 'Hold_Gray': (150, 150, 150),
}

model = YOLO("weights/best.pt")
cap = cv2.VideoCapture("videos/0505_orange.mov.mp4")
ret, frame = cap.read()
cap.release()

if not ret:
    print("❌ 첫 프레임을 읽지 못했습니다")
    exit()

frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
result = model(frame)[0]

if result.masks is None:
    print("❌ YOLO 결과에 마스크가 없습니다")
    exit()

masks = result.masks.data
boxes = result.boxes
names = model.names
img_h, img_w = frame.shape[:2]

hold_contours = []
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
        hold_contours.append((contour, color, f"{class_name} {conf:.2f}", (cx, cy)))

# 시각화 및 저장
vis_frame = frame.copy()
for contour, color, label, (cx, cy) in hold_contours:
    cv2.drawContours(vis_frame, [contour], -1, color, 2)
    x, y, w, h = cv2.boundingRect(contour)
    cv2.putText(vis_frame, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    cv2.circle(vis_frame, (cx, cy), 4, (255 - color[0], 255 - color[1], 255 - color[2]), -1)

os.makedirs("static", exist_ok=True)
cv2.imwrite(f'static/{selected_color}.jpg', vis_frame)
print(f"✅ {selected_color} 이미지 저장 완료 → static/{selected_color}.jpg")
