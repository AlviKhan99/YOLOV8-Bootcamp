from ultralytics import YOLO

# Load a pretrained YOLOv8n model
model = YOLO('yolov8n.pt')
# model = YOLO('best.pt')

# Run inference on the source
# results = model(source='0', show=True, conf=0.4) # Live inferencing

# results = model(source='0', show=True, conf=0.4, save=True) # If you want to save the live inference in your device

# results = model(source='bus.jpg', show=True, conf=0.4)

results = model(source='bus.jpg', show=True, conf=0.4, save=True) # if you want to save the inference in your device

print(results)