from ultralytics import YOLO
import multiprocessing

def main():
    # Use LAST.PT not BEST.PT for resuming
    model_path = "./runs/detect/train/last.pt"  # Path to the last checkpoint
    # Ensure this path points to your dataset YAML file
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