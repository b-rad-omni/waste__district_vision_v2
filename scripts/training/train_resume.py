from ultralytics import YOLO
import multiprocessing

def main():
    # Use LAST.PT not BEST.PT for resuming
    model_path = "C:/Users/Brad/Documents/GitHub/waste-district-vision/runs/detect/yolov8_autopipe_v5-23_merged_small_optimizedv22/weights/last.pt"
    data_yaml = "./dataset.yaml"
    
    model = YOLO(model_path)
    
    # Resume training with more epochs and higher patience
    model.train(
        data=data_yaml,
        epochs=50,              # Target more total epochs
        imgsz=1920,
        patience=20,            # Much higher patience
        resume=True             # This will continue from epoch 20
    )

if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()