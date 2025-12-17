from ultralytics import YOLO
import cv2
from google.colab.patches import cv2_imshow # Import the Colab patch

# Load model
model = YOLO('runs/detect/train/weights/best.pt')


video_path = "/content/drive/MyDrive/Colab Notebooks/test.mp4"
output_path = "detected_video.mp4"

# Mở video sử dụng open cv
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("Lỗi khi mở video")
    exit()

# Lấy thông số video để ghi kết quả
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

# Xử lý từng frame
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Dự đoán
    results = model(frame)

    # Vẽ kết quả lên frame
    # Assuming results contains at least one result object
    if results:
        annotated_frame = results[0].plot()
    else:
        annotated_frame = frame # Use original frame if no detections


    # Ghi frame đã được xử lý
    out.write(annotated_frame)

    # Hiển thị video trong lúc xử lý (tùy chọn)
    # Use cv2_imshow instead of cv2.imshow
    cv2_imshow(annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Giải phóng tài nguyên
cap.release()
out.release()
cv2.destroyAllWindows()
print("Xử lý video hoàn tất!")
