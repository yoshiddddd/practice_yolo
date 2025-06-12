# gaze_estimator.py

import mediapipe as mp
import math

class GazeEstimator:
    def __init__(self, static_mode, complexity, min_confidence):
        """MediaPipe Poseの初期化"""
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=static_mode,
            model_complexity=complexity,
            min_detection_confidence=min_confidence
        )

    def process_image(self, image_rgb):
        """画像から骨格を検出する"""
        return self.pose.process(image_rgb)

    def get_gaze_parameters(self, landmarks, image_shape):
        """ランドマークから視線推定に必要なパラメータ（目と耳の座標、角度）を計算する"""
        h, w = image_shape

        # 画面に映っている方の目と耳を主に使う
        left_eye = landmarks[self.mp_pose.PoseLandmark.LEFT_EYE_INNER]
        right_eye = landmarks[self.mp_pose.PoseLandmark.RIGHT_EYE_INNER]
        left_ear = landmarks[self.mp_pose.PoseLandmark.LEFT_EAR]
        right_ear = landmarks[self.mp_pose.PoseLandmark.RIGHT_EAR]

        eye_landmark = left_eye if left_eye.visibility > right_eye.visibility else right_eye
        ear_landmark = left_ear if left_ear.visibility > right_ear.visibility else right_ear

        # 目と耳のピクセル座標を計算
        eye_pos = (int(eye_landmark.x * w), int(eye_landmark.y * h))
        ear_pos = (int(ear_landmark.x * w), int(ear_landmark.y * h))

       #TODO ここの計算ロジック確認
        delta_x = eye_pos[0] - ear_pos[0]
        delta_y = eye_pos[1] - ear_pos[1]
        angle_rad = math.atan2(delta_y, delta_x)

        return eye_pos, ear_pos, angle_rad

    def find_target_shelf(self, eye_pos, angle_rad, shelves, image_shape, steps, x_offset):
        h, w = image_shape
        current_x, current_y = float(eye_pos[0]), float(eye_pos[1])
        
        cos_angle = math.cos(angle_rad)
        sin_angle = math.sin(angle_rad)

        for _ in range(steps):
            current_x += cos_angle
            current_y += sin_angle

            # 画面外に出たら終了→ここは画角がブレると正しく判定できない
            if not (0 < current_x < w and 0 < current_y < h):
                return "棚の外", None

            # 各棚との衝突判定
            for shelf in shelves:
                is_in_y_range = shelf["top_y"] < current_y < shelf["bottom_y"]
                is_in_x_range = (w - x_offset) < current_x < w
                
                if is_in_y_range and is_in_x_range:
                    return shelf["name"], shelf
        
        return "棚の外", None

    def close(self):
        """リソースを解放する"""
        self.pose.close()