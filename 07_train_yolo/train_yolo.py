import os
import yaml # type: ignore
import torch # type: ignore
from pathlib import Path
def check_requirements():
    try:
        import ultralytics # type: ignore
        print(f"✓ Ultralytics YOLO version: {ultralytics.__version__}")
        return True
    except ImportError:
        print("✗ Ultralytics not installed")
        print("Install it using:")
        print("  pip install ultralytics")
        return False
def train_yolo_model(data_yaml,model='yolov8n.pt',epochs=100,batch_size=16,img_size=640,project='runs/train',name='mug_6d_pose',device='0'):
    from ultralytics import YOLO # type: ignore

    # Resolve absolute path to yaml
    data_yaml = os.path.abspath(data_yaml)

    # Check YAML exists
    if not os.path.exists(data_yaml):
        print(f"Error: Data configuration file not found:\n{data_yaml}")
        return None

    # Load YAML
    with open(data_yaml, 'r') as f:
        data_config = yaml.safe_load(f)

    dataset_path = Path(data_config['path']).resolve()
    if not dataset_path.exists():
        print(f"Error: Dataset directory not found:\n{dataset_path}")
        return None

    print("\n" + "=" * 60)
    print("YOLO Training Configuration")
    print("=" * 60)
    print(f"Model        : {model}")
    print(f"Data YAML    : {data_yaml}")
    print(f"Dataset Path : {dataset_path}")
    print(f"Epochs       : {epochs}")
    print(f"Batch size   : {batch_size}")
    print(f"Image size   : {img_size}")
    print(f"Device       : {device}")
    print(f"Output dir   : {project}/{name}")
    print("=" * 60)

    # Device check
    if device != 'cpu' and not torch.cuda.is_available():
        print("CUDA not available — switching to CPU")
        device = 'cpu'

    if device != 'cpu':
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("Using CPU")

    # Initialize model
    print("\nInitializing YOLO model...")
    yolo = YOLO(model)

    # Train
    print("\nStarting training...\n")
    results = yolo.train(data=data_yaml,epochs=epochs,batch=batch_size,imgsz=img_size,project=project, name=name, device=device, patience=50,
        save=True,save_period=10,cache=False,workers=8,plots=True,verbose=True)

    print("\n" + "=" * 60)
    print("Training Complete")
    print("=" * 60)

    return results
    #validate model
def validate_model(model_path, data_yaml, img_size=640):
    from ultralytics import YOLO # type: ignore

    print("\n" + "=" * 60)
    print("Model Validation")
    print("=" * 60)

    model = YOLO(model_path)
    results = model.val(data=data_yaml, imgsz=img_size)

    print(f"mAP50     : {results.box.map50:.4f}")
    print(f"mAP50-95  : {results.box.map:.4f}")

    return results
#main
if __name__ == "__main__":

    if not check_requirements():
        exit(1)

    #config
    CONFIG = {'data_yaml': "07_train_yolo/mug.yaml",'model': 'yolov8n.pt', 'epochs': 100, 'batch_size': 16, 'img_size': 640,'project': 'runs/train',
        'name': 'mug_6d_pose','device': '0'}
    print("\n" + "=" * 60)
    print("Mug 6D Pose Estimation – YOLO Training")
    print("=" * 60)
    print("This will train a YOLO model to detect mugs.\n")

    response = input("Start training? (yes/no): ").lower()
    if response not in ['yes', 'y']:
        print("Training cancelled.")
        exit(0)

    # Train
    results = train_yolo_model(**CONFIG)

    # Validate best model
    if results:
        # Use pathlib with explicit string casts to satisfy strict linter
        project_dir = str(CONFIG['project'])
        run_name = str(CONFIG['name'])
        best_model_path = Path(project_dir) / run_name / 'weights' / 'best.pt'
        best_model = str(best_model_path)
        if os.path.exists(best_model):
            print("\nRunning validation on best model...")
            validate_model(best_model, CONFIG['data_yaml'])
        else:
            print("best.pt not found — training may not have completed fully.")
