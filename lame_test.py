from ultralytics import YOLO

# Load your custom trained model
model = YOLO('runs/detect/runs/detect/droneRTX/weights/best.pt') 

# Validate on the test set
metrics = model.val(data='data.yaml', split='test')

# Print the key metric judges care about
print(f"mAP50: {metrics.box.map50}")
print(f"mAP50-95: {metrics.box.map}")