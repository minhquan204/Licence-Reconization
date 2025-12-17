from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt

# Load model tốt nhất để test ảnh
model = YOLO('runs/detect/train/weights/best.pt')


image_path = "/content/drive/MyDrive/Colab Notebooks/Ảnh/IMG_0033.JPG"

# Dự đoán
results = model.predict(
    source=image_path,
    conf=0.25,  # Ngưỡng tin cậy
    save=True,   # Lưu ảnh kết quả
    show_labels=True,
    show_conf=True
)

# Hiển thị ảnh kết quả trực tiếp trên Colab
for r in results:
    im_array = r.plot()  # Vẽ bounding boxes lên ảnh
    im_rgb = cv2.cvtColor(im_array, cv2.COLOR_BGR2RGB)
    plt.imshow(im_rgb)
    plt.axis('off')
    plt.show()
