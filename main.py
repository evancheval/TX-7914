from ultralytics import YOLO

model_source="models\\yolo26n.pt"

model = YOLO(model_source)

results = model.predict(source="data\\TD_DIO5_Seance2_Box4_Groupe1_Part1_Up_left.mp4", show=True)