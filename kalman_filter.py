"""
HIAS Gimbal System - 2D Kalman Filter Module
"""
import cv2
import numpy as np

class TargetTrackerKF:
    def __init__(self):
        self.kf = cv2.KalmanFilter(4, 2)
        self.kf.transitionMatrix = np.array(
            [[1, 0, 1, 0],[0, 1, 0, 1],[0, 0, 1, 0],[0, 0, 0, 1]], np.float32)
        self.kf.measurementMatrix = np.array(
            [[1, 0, 0, 0],[0, 1, 0, 0]], np.float32)
        self.kf.processNoiseCov = np.eye(4, dtype=np.float32) * 1e-2
        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 1e-1
        self.initialized = False

    def predict_and_update(self, current_x, current_y, is_target_lost):
        predicted = self.kf.predict()
        pred_x, pred_y = int(predicted[0][0]), int(predicted[1][0])

        if not is_target_lost:
            measured = np.array([[np.float32(current_x)], [np.float32(current_y)]])
            if not self.initialized:
                self.kf.statePre = np.array([[measured[0][0]], [measured[1][0]],[0], [0]], np.float32)
                self.kf.statePost = np.array([[measured[0][0]], [measured[1][0]], [0], [0]], np.float32)
                self.initialized = True
            self.kf.correct(measured)
            return current_x, current_y, pred_x, pred_y
        else:
            self.initialized = False
            return pred_x, pred_y, pred_x, pred_y
