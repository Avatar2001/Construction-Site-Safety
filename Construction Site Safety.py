from ultralytics import YOLO

# Initialize YOLO model with the best.pt checkpoint
model = YOLO('best.pt')

# Make predictions on an image
results = model.predict("th (2).jpeg", imgsz=640, conf=0.3, save=True, show=True)

# Display the results
print(results)
