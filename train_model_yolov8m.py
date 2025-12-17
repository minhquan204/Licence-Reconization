# retrain với model tốt hơn (yolov8m)
def advanced_training():
    from roboflow import Roboflow
    from ultralytics import YOLO
    import torch

    # Kiểm tra GPU
    print(f"GPU available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Tải dataset từ Roboflow
    rf = Roboflow(api_key="hyWj56r4YyxOfvHC1BIZ")
    project = rf.workspace("vietnameselicenseplate").project("vietnamese-license-plate-tptd0-y4gwu")

    # Tải dataset với augmentations
    dataset = project.version(1).download("yolov8")

    # Load model YOLOv8
    model = YOLO("yolov8m.pt")

    # Training với hyperparameters tối ưu
    results = model.train(
        data=f"{dataset.location}/data.yaml",
        epochs=100,                    # Tăng epochs
        imgsz=640,
        batch=16,
        lr0=0.01,                      # Learning rate
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3.0,
        patience=15,                   # Early stopping
        device='cuda',
        workers=8,
        augment=True,                  # Bật augment
        hsv_h=0.015,                   # Augmentation parameters
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=10.0,                  # Xoay ảnh
        translate=0.1,                 # Dịch chuyển
        scale=0.5,                     # Scale
        shear=2.0,                     # Nghiêng
        perspective=0.001,             # Phối cảnh
        flipud=0.5,                    # Lật lên xuống
        fliplr=0.5,                    # Lật trái phải
        mosaic=1.0,                    # Mosaic augmentation
        mixup=0.1,                     # Mixup augmentation
        copy_paste=0.1,                # Copy-paste augmentation
        erasing=0.4,                   # Random erasing
        crop_fraction=0.9,             # Random crop
    )

    return results

# Chạy training nâng cao
advanced_training()
