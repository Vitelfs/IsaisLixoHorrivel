import cv2
import cvzone
import math
import requests
from ultralytics import YOLO

cap = cv2.VideoCapture('fall.mp4')
model = YOLO('yolov8s.pt')

classnames = []
with open('classes.txt', 'r') as f:
    classnames = f.read().splitlines()

POST_URL = "http://localhost:3000/enviar-mensagem"
NUMERO_ALARME = "557582949946"
MENSAGEM_QUEDA = "Alerta: Isaias caiu na pica!"

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    frame = cv2.resize(frame, (980, 740))
    results = model(frame)

    for info in results:
        parameters = info.boxes
        for box in parameters:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            confidence = box.conf[0]
            class_detect = box.cls[0]
            class_detect = int(class_detect)
            class_detect = classnames[class_detect]
            conf = math.ceil(confidence * 100)

            height = y2 - y1
            width = x2 - x1
            threshold = height - width

            if conf > 80 and class_detect == 'person':
                cvzone.cornerRect(frame, [x1, y1, width, height], l=30, rt=6)
                cvzone.putTextRect(frame, f'{class_detect}', [x1 + 8, y1 - 12], thickness=2, scale=2)
            
            if threshold < 0:
                cvzone.putTextRect(frame, 'Fall Detected', [x1, y1 - 30], thickness=2, scale=2)
                try:
                    response = requests.post(
                        POST_URL,
                        json={
                            "numero": NUMERO_ALARME,
                            "mensagem": MENSAGEM_QUEDA
                        }
                    )
                    if response.status_code == 200:
                        print("Alerta enviado com sucesso!")
                    else:
                        print(f"Falha ao enviar alerta: {response.status_code}, {response.text}")
                except Exception as e:
                    print(f"Erro ao enviar o alerta: {e}")

    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('t'):
        break

cap.release()
cv2.destroyAllWindows()
