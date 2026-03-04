"""
HIAS Gimbal System - Main Control Engine
Author: Li Guanru
"""
import os
import cv2
import time
import serial
from collections import deque
from ultralytics import YOLO

from config import *
from core.kalman_filter import TargetTrackerKF

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

def draw_hud(frame, err_x_history, err_y_history, W, H):
    graph_w, graph_h = 200, 100
    graph_x, graph_y = W - graph_w - 10, H - graph_h - 10
    overlay = frame.copy()
    cv2.rectangle(overlay, (graph_x, graph_y), (graph_x + graph_w, graph_y + graph_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
    zero_y = graph_y + graph_h // 2
    cv2.line(frame, (graph_x, zero_y), (graph_x + graph_w, zero_y), (100, 100, 100), 1)

    for i in range(1, HISTORY_LEN):
        val_x1 = int(err_x_history[i-1] * (graph_h/2) / (W/2))
        val_x2 = int(err_x_history[i] * (graph_h/2) / (W/2))
        cv2.line(frame, (graph_x + int((i-1) * (graph_w/HISTORY_LEN)), zero_y - val_x1),
                        (graph_x + int(i * (graph_w/HISTORY_LEN)), zero_y - val_x2), (255, 255, 0), 1)
        val_y1 = int(err_y_history[i-1] * (graph_h/2) / (H/2))
        val_y2 = int(err_y_history[i] * (graph_h/2) / (H/2))
        cv2.line(frame, (graph_x + int((i-1) * (graph_w/HISTORY_LEN)), zero_y - val_y1),
                        (graph_x + int(i * (graph_w/HISTORY_LEN)), zero_y - val_y2), (255, 0, 255), 1)
    cv2.putText(frame, "Telemetry: X(Cyan) Y(Purple)", (graph_x, graph_y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)

def main():
    print("🚀 Initializing HIAS-KCF Optical Tracker...")
    ser = None
    if ENABLE_SERIAL:
        try:
            ser = serial.Serial(COM_PORT, BAUD_RATE, timeout=0.01)
            print(f"✅ Serial Link Established on {COM_PORT}")
        except Exception as e:
            print(f"⚠️ Serial Link Failed: {e}")

    try:
        model = YOLO(MODEL_PATH)
    except Exception as e:
        print(f"❌ Core Model Missing: {e}")
        print(f"👉 请确保 {MODEL_PATH} 文件存在！")
        return

    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    W, H = int(cap.get(3)), int(cap.get(4))
    CENTER_X, CENTER_Y = W // 2, H // 2

    kf_tracker = TargetTrackerKF()
    err_x_hist = deque([0]*HISTORY_LEN, maxlen=HISTORY_LEN)
    err_y_hist = deque([0]*HISTORY_LEN, maxlen=HISTORY_LEN)

    print("▶️ System Ready. Entering main loop...")
    while True:
        ret, frame = cap.read()
        if not ret: break

        results = model.track(frame, conf=CONF_THRESHOLD, persist=True, verbose=False)
        raw_x, raw_y, conf_score = 0, 0, 0.0
        target_lost = True

        if results[0].boxes:
            box = results[0].boxes[0]
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf_score = box.conf[0]
            raw_x, raw_y = (x1 + x2) // 2, (y1 + y2) // 2
            target_lost = False
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        smooth_x, smooth_y, pred_x, pred_y = kf_tracker.predict_and_update(raw_x, raw_y, target_lost)

        cv2.line(frame, (CENTER_X - 20, CENTER_Y), (CENTER_X + 20, CENTER_Y), (0, 0, 255), 1)
        cv2.line(frame, (CENTER_X, CENTER_Y - 20), (CENTER_X, CENTER_Y + 20), (0, 0, 255), 1)
        cv2.circle(frame, (pred_x, pred_y), 15, (0, 255, 255), 2) # 卡尔曼预测点

        err_x, err_y, dir_x, dir_y = 0, 0, 0, 0

        if not target_lost or (pred_x > 0 and pred_y > 0):
            err_x = smooth_x - CENTER_X
            err_y = smooth_y - CENTER_Y
            cv2.line(frame, (CENTER_X, CENTER_Y), (smooth_x, smooth_y), (255, 0, 255), 2, cv2.LINE_AA)
            cv2.putText(frame, f"LOCK | KF:ON | ERR:[{err_x}, {err_y}]", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            if abs(err_x) > DEAD_ZONE: dir_x = 1 if err_x < 0 else 2
            if abs(err_y) > DEAD_ZONE: dir_y = 1 if err_y < 0 else 2

            packet = bytearray([0xFF, dir_x, min(abs(err_x), 255), dir_y, min(abs(err_y), 255), 0xFE])
            if ser and ser.is_open: ser.write(packet)
        else:
            cv2.putText(frame, "TARGET LOST - SCANNING", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        err_x_hist.append(err_x)
        err_y_hist.append(err_y)
        draw_hud(frame, err_x_hist, err_y_hist, W, H)

        cv2.imshow('HIAS-KCF Optical Tracker', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    if ser: ser.close()
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
