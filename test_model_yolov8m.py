#test lại với hình ảnh
import cv2
import easyocr
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO
import torch

class AdvancedLicensePlateRecognizer:
    def __init__(self, plate_detector_weights='/content/drive/MyDrive/Colab Notebooks/runs*/detect/train/weights/best.pt'):
        # Load model phát hiện biển số
        self.plate_model = YOLO(plate_detector_weights)

        # Khởi tạo EasyOCR cho tiếng Việt
        self.reader = easyocr.Reader(['vi', 'en'])

    def enhance_contrast(self, image):
        """Nâng cao độ tương phản sử dụng nhiều phương pháp"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        # Phương pháp 1: CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        clahe_applied = clahe.apply(gray)

        # Phương pháp 2: Gamma correction
        gamma = 1.5
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
        gamma_corrected = cv2.LUT(clahe_applied, table)

        # Phương pháp 3: Làm sắc nét
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        sharpened = cv2.filter2D(gamma_corrected, -1, kernel)

        return sharpened

    def remove_noise(self, image):
        """Loại bỏ nhiễu và làm sạch ảnh"""
        # Median blur để giảm noise
        denoised = cv2.medianBlur(image, 3)

        # Gaussian blur nhẹ
        gaussian = cv2.GaussianBlur(denoised, (1, 1), 0)

        return gaussian

    def adaptive_thresholding(self, image):
        """Áp dụng adaptive thresholding"""
        # Adaptive threshold
        binary = cv2.adaptiveThreshold(
            image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 11, 2
        )

        # Morphological operations để làm sạch
        kernel = np.ones((2,2), np.uint8)
        cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)

        return cleaned

    def advanced_preprocess_plate(self, plate_image):
        """Tiền xử lý nâng cao cho ảnh biển số"""
        if plate_image is None or plate_image.size == 0:
            return None

        #Tăng cường độ tương phản
        contrast_enhanced = self.enhance_contrast(plate_image)

        #Loại bỏ nhiễu
        noise_removed = self.remove_noise(contrast_enhanced)

        #Resize ảnh để cải thiện OCR
        height, width = noise_removed.shape
        if height > 0 and width > 0:
            # Tăng kích thước để cải thiện chất lượng
            scale_factor = 100.0 / height
            new_width = int(width * scale_factor)
            if new_width > 0:
                resized = cv2.resize(noise_removed, (new_width, 100))
            else:
                resized = noise_removed
        else:
            resized = noise_removed

        #Adaptive thresholding
        thresholded = self.adaptive_thresholding(resized)

        return thresholded

    def detect_license_plate(self, image_path, conf_threshold=0.5):
        """Phát hiện và cắt biển số từ ảnh"""
        # Đọc ảnh
        image = cv2.imread(image_path)
        if image is None:
            print("Không thể đọc ảnh")
            return None, None, None

        # Dự đoán biển số
        results = self.plate_model(image, conf=conf_threshold)

        plates = []
        bboxes = []

        for result in results:
            if result.boxes is not None and len(result.boxes) > 0:
                for box in result.boxes:
                    # Lấy tọa độ bounding box
                    x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                    confidence = box.conf[0].cpu().numpy()

                    # Cắt biển số
                    plate_image = image[y1:y2, x1:x2]

                    plates.append(plate_image)
                    bboxes.append((x1, y1, x2, y2, confidence))

        return plates, bboxes, image

    def recognize_characters(self, plate_image):
        """Nhận diện ký tự trên biển số với tiền xử lý nâng cao"""
        if plate_image is None or plate_image.size == 0:
            return "", 0, []

        # Tiền xử lý nâng cao ảnh biển số
        processed_plate = self.advanced_preprocess_plate(plate_image)

        if processed_plate is None:
            return "", 0, []

        # Nhận diện ký tự với EasyOCR
        try:
            # Thử với các tham số khác nhau để tối ưu
            results = self.reader.readtext(processed_plate,
                                         detail=1,
                                         paragraph=False,
                                         width_ths=0.8,  # Tăng độ linh hoạt về chiều rộng
                                         height_ths=0.8, # Tăng độ linh hoạt về chiều cao
                                         min_size=20,    # Tăng kích thước tối thiểu
                                         text_threshold=0.4,  # Giảm ngưỡng text
                                         low_text=0.3)   # Giảm ngưỡng low text
        except Exception as e:
            print(f"Lỗi OCR: {e}")
            return "", 0, []

        characters = []
        total_confidence = 0

        for (bbox, text, confidence) in results:
            # Lọc và làm sạch text
            cleaned_text = ''.join(filter(lambda x: x.isalnum() or x in '- ', text)).upper()
            cleaned_text = cleaned_text.replace(' ', '')

            if len(cleaned_text) > 0 and confidence > 0.2:  # Giảm ngưỡng confidence
                characters.append({
                    'text': cleaned_text,
                    'confidence': confidence,
                    'bbox': bbox
                })
                total_confidence += confidence

        # Sắp xếp ký tự từ trái sang phải
        if characters:
            characters.sort(key=lambda x: np.mean([point[0] for point in x['bbox']]))

        # Ghép các ký tự thành biển số hoàn chỉnh
        license_text = ''.join([char['text'] for char in characters])
        avg_confidence = total_confidence / len(characters) if characters else 0

        return license_text, avg_confidence, characters

    def detect_license_plate_from_frame(self, frame, conf_threshold=0.5):
        """Phát hiện biển số từ frame video"""
        results = self.plate_model(frame, conf=conf_threshold)

        plates = []
        bboxes = []

        for result in results:
            if result.boxes is not None and len(result.boxes) > 0:
                for box in result.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                    confidence = box.conf[0].cpu().numpy()

                    plate_image = frame[y1:y2, x1:x2]

                    if plate_image.size > 0 and plate_image.shape[0] > 10 and plate_image.shape[1] > 10:
                        plates.append(plate_image)
                        bboxes.append((x1, y1, x2, y2, confidence))

        return plates, bboxes

def test_with_preview(image_path):
    """Test với xem trước các bước tiền xử lý"""
    recognizer = AdvancedLicensePlateRecognizer()

    # Phát hiện và cắt biển số
    plates, bboxes, original_image = recognizer.detect_license_plate(image_path)

    if not plates:
        print("Không phát hiện được biển số")
        return

    print(f"Phát hiện được {len(plates)} biển số")

    for i, (plate, bbox) in enumerate(zip(plates, bboxes)):
        print(f"\n--- Xử lý biển số {i+1} ---")

        # Hiển thị các bước tiền xử lý
        fig, axes = plt.subplots(2, 3, figsize=(15, 8))

        # Ảnh gốc
        axes[0,0].imshow(cv2.cvtColor(plate, cv2.COLOR_BGR2RGB))
        axes[0,0].set_title('Biển số gốc')
        axes[0,0].axis('off')

        # Tăng cường độ tương phản
        contrast_enhanced = recognizer.enhance_contrast(plate)
        axes[0,1].imshow(contrast_enhanced, cmap='gray')
        axes[0,1].set_title('Sau tăng tương phản')
        axes[0,1].axis('off')

        # Loại bỏ nhiễu
        noise_removed = recognizer.remove_noise(contrast_enhanced)
        axes[0,2].imshow(noise_removed, cmap='gray')
        axes[0,2].set_title('Sau khử nhiễu')
        axes[0,2].axis('off')

        # Adaptive thresholding
        thresholded = recognizer.adaptive_thresholding(noise_removed)
        axes[1,0].imshow(thresholded, cmap='gray')
        axes[1,0].set_title('Sau adaptive threshold')
        axes[1,0].axis('off')

        # Ảnh cuối cùng sau xử lý
        final_processed = recognizer.advanced_preprocess_plate(plate)
        axes[1,1].imshow(final_processed, cmap='gray')
        axes[1,1].set_title('Ảnh cuối cùng')
        axes[1,1].axis('off')

        # Nhận diện ký tự
        license_text, confidence, characters = recognizer.recognize_characters(plate)

        # Hiển thị kết quả
        axes[1,2].imshow(final_processed, cmap='gray')
        axes[1,2].set_title(f'Kết quả: {license_text}\nĐộ tin cậy: {confidence:.2f}')
        axes[1,2].axis('off')

        plt.tight_layout()
        plt.show()

        print(f"Biển số {i+1}")


        # Thử với các phương pháp OCR khác nhau
        print("\nThử nghiệm với các phương pháp khác:")
        test_different_ocr_methods(recognizer, plate)

def test_different_ocr_methods(recognizer, plate_image):
    """Thử các phương pháp OCR khác nhau"""

    results1 = recognizer.reader.readtext(plate_image, detail=1, paragraph=False)
    text1 = ''.join([res[1] for res in results1 if len(res) > 2 and res[2] > 0.3])
    print(f"Kết quả dự đoán: {text1}")


# Chạy test với ảnh
test_with_preview("/content/drive/MyDrive/Colab Notebooks/Ảnh/z7121409651831_6324aee054f8781e7cbb9ed50bfd85c6.jpg")
