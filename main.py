from ultralytics import YOLO
import sys

SHOW = False
SAVE = True

if len(sys.argv) < 2:
    print("Usage: python main.py <media_path>")
    sys.exit(1)

file_path = sys.argv[1]

if len(sys.argv) > 2:
    model_source = sys.argv[2]
else:
    model_source="models/yolo26n.pt"

model = YOLO(model_source)

try : 
    model.overrides['classes'] = 0
    results = model.track(source=file_path, show=SHOW, save=SAVE, verbose=False)
except KeyboardInterrupt:
    print("Process interrupted by user.")