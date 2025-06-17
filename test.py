# import cv2
# import mediapipe as mp
# import numpy as np

# mp_pose = mp.solutions.pose
# mp_drawing = mp.solutions.drawing_utils
# mp_drawing_styles = mp.solutions.drawing_styles

# image = cv2.imread("input-img/no-ear2.jpeg")
# if image is None:
#     print("画像を読み込めません")
#     exit(1)
# else:
#     print(f"画像を読み込みました: {image.shape}")
# with mp_pose.Pose(
#         static_image_mode=True,
#         model_complexity=2,
#         enable_segmentation=False,
#         min_detection_confidence=0.6
#     ) as pose:
#     rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#     results = pose.process(rgb_image)
#     annotated_image = image.copy()
#     if results.pose_landmarks:
#         print("骨格が検出されました！")
#         mp_drawing.draw_landmarks(
#             annotated_image,
#             results.pose_landmarks,
#             mp_pose.POSE_CONNECTIONS,
#             landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
#         )
#     else:
#         print("骨格が検出されませんでした")
#     cv2.imshow("Original Image", annotated_image)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
import cv2
import mediapipe as mp
import numpy as np

# MediaPipeのモジュールを初期化
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# 画像を読み込む
# ご自身の画像ファイルパスに変更してください
image = cv2.imread("input-img/three.jpeg") 
if image is None:
    print("画像を読み込めません")
    exit(1)
else:
    print(f"画像を読み込みました: {image.shape}")
    image_height, image_width, _ = image.shape

# Poseモデルを初期化
with mp_pose.Pose(
        static_image_mode=True,
        model_complexity=2,
        enable_segmentation=False,
        min_detection_confidence=0.5
    ) as pose:

    # BGR画像をRGBに変換
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # 骨格検出を実行
    results = pose.process(rgb_image)

    # 描画用の画像をコピー
    annotated_image = image.copy()

    # 骨格が検出された場合
    if results.pose_landmarks:
        print("骨格が検出されました！")
        
        # --- ▼ここからが追加部分▼ ---

        # 1. 肩のランドマークを取得
        landmarks = results.pose_landmarks.landmark
        left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
        right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]

        # 2. 両肩が検出されているか確認 (visibilityで信頼度をチェック)
        if left_shoulder.visibility > 0.5 and right_shoulder.visibility > 0.5:
            
            # 3. 左右の肩の中点を計算 (これが首の座標)
            neck_x = (left_shoulder.x + right_shoulder.x) / 2
            neck_y = (left_shoulder.y + right_shoulder.y) / 2

            print("--- 首の座標 ---")
            # 正規化された座標 (0.0 ~ 1.0)
            print(f"正規化座標: (x={neck_x:.4f}, y={neck_y:.4f})")
            
            # 画像上のピクセル座標に変換
            pixel_neck_x = int(neck_x * image_width)
            pixel_neck_y = int(neck_y * image_height)
            print(f"ピクセル座標: (x={pixel_neck_x}, y={pixel_neck_y})")

            # 4. 計算した首の位置に円を描画
            cv2.circle(
                annotated_image,               # 描画対象の画像
                (pixel_neck_x, pixel_neck_y),  # 円の中心座標 (ピクセル)
                10,                            # 円の半径
                (0, 255, 0),                   # 色 (B, G, R) -> 緑
                -1                             # 塗りつぶし
            )
        else:
            print("肩のランドマークが検出できなかったため、首の座標は計算できませんでした。")

        # --- ▲ここまでが追加部分▲ ---

        # 元のコードのランドマーク描画も残す場合
        mp_drawing.draw_landmarks(
            annotated_image,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2),
            connection_drawing_spec=mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
        )

    else:
        print("骨格が検出されませんでした")

    # 結果の画像を表示
    cv2.imshow("Neck Position", annotated_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()