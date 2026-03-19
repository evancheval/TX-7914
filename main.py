from ultralytics import YOLO
import sys


if len(sys.argv) < 2:
    print("Usage: python main.py <media_path>")
    sys.exit(1)

file_path = sys.argv[1]

model_source="models/yolo26n.pt"

model = YOLO(model_source)

try : 
    model.overrides['classes'] = 0
    results = model.predict(source=file_path, show=True)
except KeyboardInterrupt:
    print("Process interrupted by user.")