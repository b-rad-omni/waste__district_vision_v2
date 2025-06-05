from ultralytics import YOLO
import multiprocessing

def main():
    model_path = "change to your local model path"  # e.g., "./yolo_model.pt"
    # Ensure the model path is correct
    data_yaml = "./dataset.yaml"
    
    model = YOLO(model_path)
    
    # Run validation with more detailed output
    results = model.val(
        data=data_yaml,
        imgsz=1920,
        save_json=True,     # Save JSON results
        save_hybrid=True,   # Save hybrid results
        plots=True,         # Generate plots
        verbose=True        # Verbose output
    )
    
    print("Validation complete!")
    print(f"Results saved to: {results.save_dir}")

if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()