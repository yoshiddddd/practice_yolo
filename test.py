import cv2
import mediapipe as mp
import numpy as np

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

image = cv2.imread("input-img/test.jpeg")
if image is None:
    print("画像を読み込めません")
    exit(1)
else:
    print(f"画像を読み込みました: {image.shape}")
with mp_pose.Pose(
        static_image_mode=True,
        model_complexity=2,
        enable_segmentation=False,
        min_detection_confidence=0.6
    ) as pose:
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb_image)
    annotated_image = image.copy()
    if results.pose_landmarks:
        print("骨格が検出されました！")
        mp_drawing.draw_landmarks(
            annotated_image,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
        )
    else:
        print("骨格が検出されませんでした")
    cv2.imshow("Original Image", annotated_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()