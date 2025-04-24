import cv2
import math
import time
import numpy as np
from ultralytics import YOLO
from threading import Thread, Lock
from queue import Queue
import requests
import os
import json
from flask import Flask, jsonify, Response

app = Flask(__name__)

class FallDetection:
    def __init__(self, video_source='fall.mp4', model_path='yolov8s.pt', classes_path='classes.txt',
                 conf_threshold=0.6, alert_cooldown=30, target_fps=30):
        self.cap = cv2.VideoCapture(video_source)
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        self.running = True
        self.alert_sent_time = 0
        self.alert_cooldown = alert_cooldown
        self.fall_frames_counter = 0
        self.normal_frames_counter = 0
        self.fall_threshold = 2
        self.recovery_threshold = 5
        self.fall_status = False
        self.original_fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.target_fps = min(target_fps, self.original_fps)
        self.frame_delay = 1.0 / self.target_fps
        self.original_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.original_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.POST_URL = "http://localhost:3000/enviar-mensagem"
        self.NUMERO_ALARME = "557582949946"
        self.MENSAGEM_QUEDA = "Alerta: Foi detectada uma queda!"
        try:
            with open(classes_path, 'r') as f:
                self.classnames = f.read().splitlines()
        except FileNotFoundError:
            self.classnames = ["person"]
        self.display_width = min(960, self.original_width)
        self.display_height = int(self.display_width * self.original_height / self.original_width)
        self.frames_queue = Queue(maxsize=2)
        self.results_queue = Queue(maxsize=2)
        self.detection_thread = Thread(target=self.detection_loop)
        self.detection_thread.daemon = True
        self.current_frame = None
        self.frame_lock = Lock()
        self.last_processed_frame = None
        self.last_frame_time = 0
        self.last_fall_time = 0
        self.min_fall_display_time = 2.0
        self.current_frame_fall_detected = False
        self.detection_thread.start()
        self.processing_thread = Thread(target=self.process_video)
        self.processing_thread.daemon = True
        self.processing_thread.start()

    def send_alert(self):
        try:
            payload = {
                "numero": self.NUMERO_ALARME,
                "mensagem": self.MENSAGEM_QUEDA
            }
            requests.post(self.POST_URL, json=payload, timeout=5)
            print("Alerta enviado com sucesso!")
        except Exception as e:
            print(f"Erro ao enviar alerta: {e}"
            )

    def detection_loop(self):
        while self.running:
            if not self.frames_queue.empty():
                original_frame = self.frames_queue.get()
                try:
                    results = self.model(original_frame, 
                                        conf=self.conf_threshold, 
                                        classes=0,
                                        verbose=False)
                    self.results_queue.put((original_frame, results[0]))
                except Exception as e:
                    print(f"Erro na detecção: {e}")
                    self.results_queue.put((original_frame, None))
            else:
                time.sleep(0.001)

    def process_detection_results(self, frame, results):
        current_fall_detected = False
        output_frame = frame.copy()
        if results is not None and hasattr(results, 'boxes') and results.boxes:
            for box in results.boxes:
                try:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    confidence = float(box.conf[0])
                    height = y2 - y1
                    width = x2 - x1
                    aspect_ratio = height / max(width, 1)
                    area = width * height
                    is_fall = aspect_ratio < 1.3 and width > 35 and area > 4000
                    if confidence > self.conf_threshold:
                        thickness = max(1, min(3, int(area / 10000)))
                        box_color = (0, 0, 255) if is_fall else (0, 255, 0)
                        cv2.rectangle(output_frame, (x1, y1), (x2, y2), box_color, thickness)
                        conf_text = f'Pessoa {int(confidence*100)}%'
                        text_size, _ = cv2.getTextSize(conf_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                        cv2.rectangle(output_frame, (x1, y1-text_size[1]-5), (x1+text_size[0], y1), box_color, -1)
                        cv2.putText(output_frame, conf_text, (x1, y1-5), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
                        if is_fall:
                            current_fall_detected = True
                            fall_text = 'QUEDA!'
                            text_size, _ = cv2.getTextSize(fall_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                            cv2.rectangle(output_frame, (x1, y1-text_size[1]-30), (x1+text_size[0], y1-25), (0, 0, 255), -1)
                            cv2.putText(output_frame, fall_text, (x1, y1-30), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                except Exception as e:
                    print(f"Erro ao processar detecção: {e}")
                    continue
        self.current_frame_fall_detected = current_fall_detected
        return output_frame, current_fall_detected

    def update_fall_status(self, current_fall_detected):
        current_time = time.time()
        if current_fall_detected:
            self.fall_frames_counter += 1
            self.normal_frames_counter = 0
            if self.fall_frames_counter == self.fall_threshold:
                self.last_fall_time = current_time
        else:
            self.normal_frames_counter += 1
            self.fall_frames_counter = max(0, self.fall_frames_counter - 0.5)
        if not self.fall_status and self.fall_frames_counter >= self.fall_threshold:
            self.fall_status = True
            print("Queda detectada!")
            return True
        elif self.fall_status and self.normal_frames_counter >= self.recovery_threshold:
            time_since_last_fall = current_time - self.last_fall_time
            if time_since_last_fall >= self.min_fall_display_time:
                self.fall_status = False
                print("Pessoa recuperada")
        return False

    def get_frame(self):
        with self.frame_lock:
            if self.current_frame is not None:
                try:
                    ret, jpeg = cv2.imencode('.jpg', self.current_frame)
                    return jpeg.tobytes()
                except Exception as e:
                    print(f"Erro ao codificar frame: {e}")
                    return None
            return None

    def get_status(self):
        return {
            'fall_detected': self.fall_status,
            'current_frame_fall': self.current_frame_fall_detected,
            'confidence': min(1.0, self.fall_frames_counter / self.fall_threshold) if self.fall_threshold > 0 else 0,
            'frames_counter': self.fall_frames_counter,
            'threshold': self.fall_threshold
        }

    def process_video(self):
        while self.running and self.cap.isOpened():
            current_time = time.time()
            if current_time - self.last_frame_time >= self.frame_delay:
                ret, frame = self.cap.read()
                if not ret:
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                self.last_frame_time = current_time
                if not self.frames_queue.full():
                    self.frames_queue.put(frame)
                if not self.results_queue.empty():
                    frame_to_draw, detection_results = self.results_queue.get()
                    output_frame, current_fall_detected = self.process_detection_results(frame_to_draw, detection_results)
                    new_fall = self.update_fall_status(current_fall_detected)
                    if new_fall:
                        now = time.time()
                        if now - self.alert_sent_time > self.alert_cooldown:
                            self.alert_sent_time = now
                            Thread(target=self.send_alert).start()
                    resized = cv2.resize(output_frame, (self.display_width, self.display_height))
                    with self.frame_lock:
                        self.current_frame = resized
            else:
                time.sleep(0.001)

fall_detection = FallDetection()

@app.route('/video_feed')
def video_feed():
    def generate():
        while True:
            frame = fall_detection.get_frame()
            if frame is not None:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            else:
                time.sleep(0.1)
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/status')
def status():
    return jsonify(fall_detection.get_status())

if __name__ == '__main__':
    try:
        print("API iniciada em http://127.0.0.1:5000")
        app.run(host='0.0.0.0', port=5000, debug=False)
    except Exception as e:
        print(f"Erro ao iniciar o servidor: {e}")
    finally:
        if detector is not None:
            detector.stop()