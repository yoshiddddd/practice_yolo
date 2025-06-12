# config.py

# --- 棚の座標設定 (画像の高さに応じて調整) ---
# 元画像の高さが約1000pxだったので、それに合わせています。
# 使用する画像サイズに合わせて調整してください。
SHELF_BOUNDARIES = [
    {"name": "1段目", "top_y": 520, "bottom_y": 585},
    {"name": "2段目", "top_y": 585, "bottom_y": 670},
    {"name": "3段目", "top_y": 670, "bottom_y": 755},
    {"name": "4段目", "top_y": 755, "bottom_y": 850},
]

# --- フォントファイルの設定 ---
FONT_PATH = 'ipaexg.ttf'  # 日本語フォントファイル
FONT_SIZE = 40

# --- MediaPipe Poseモデルの設定 ---
MODEL_COMPLEXITY = 1
MIN_DETECTION_CONFIDENCE = 0.5
STATIC_IMAGE_MODE = True

# --- 視線推定のパラメータ ---
GAZE_PROJECTION_STEPS = 1500 # 視線をどこまで延長して探索するか
SHELF_DETECTION_X_OFFSET = 20 # 棚の右端からどのくらいの範囲を当たり判定とみなすか