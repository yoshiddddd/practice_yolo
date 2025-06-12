import cv2
import mediapipe as mp
import numpy as np

# MediaPipeの初期化
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

def detect_pose_from_image(image_path, output_path=None):
    """
    静止画像から骨格を検出し、結果を表示・保存する
    
    Args:
        image_path (str): 入力画像のパス
        output_path (str, optional): 結果画像の保存パス
    """
    # 画像の読み込み
    image = cv2.imread(image_path)
    if image is None:
        print(f"エラー: 画像を読み込めません - {image_path}")
        return None
    
    print(f"画像を読み込みました: {image.shape}")
    
    # MediaPipe Poseモデルの初期化
    with mp_pose.Pose(
        static_image_mode=True,           # 静止画モード
        model_complexity=2,               # 高精度モデル
        enable_segmentation=False,        # セグメンテーション無効
        min_detection_confidence=0.5      # 検出の最小信頼度
    ) as pose:
        
        # BGRからRGBに変換
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 骨格検出の実行
        results = pose.process(rgb_image)
        
        # 結果をコピーして描画用に準備
        annotated_image = image.copy()
        
        if results.pose_landmarks:
            print("骨格が検出されました！")
            
            # 骨格の描画
            mp_drawing.draw_landmarks(
                annotated_image,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
            )
            
            # キーポイント情報の表示
            print_landmark_info(results.pose_landmarks, image.shape)
            
        else:
            print("骨格が検出されませんでした")
            return None
        
        # 結果の表示
        cv2.imshow('Original Image', image)
        cv2.imshow('Pose Detection Result', annotated_image)
        
        # 結果画像の保存
        if output_path:
            cv2.imwrite(output_path, annotated_image)
            print(f"結果画像を保存しました: {output_path}")
        
        print("何かキーを押すと終了します...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        return results

def print_landmark_info(pose_landmarks, image_shape):
    """
    主要なランドマークの座標情報を表示
    """
    landmarks = pose_landmarks.landmark
    h, w, _ = image_shape
    
    # 主要キーポイントの座標を表示
    key_points = {
        'NOSE': mp_pose.PoseLandmark.NOSE,
        'LEFT_SHOULDER': mp_pose.PoseLandmark.LEFT_SHOULDER,
        'RIGHT_SHOULDER': mp_pose.PoseLandmark.RIGHT_SHOULDER,
        'LEFT_ELBOW': mp_pose.PoseLandmark.LEFT_ELBOW,
        'RIGHT_ELBOW': mp_pose.PoseLandmark.RIGHT_ELBOW,
        'LEFT_WRIST': mp_pose.PoseLandmark.LEFT_WRIST,
        'RIGHT_WRIST': mp_pose.PoseLandmark.RIGHT_WRIST,
        'LEFT_HIP': mp_pose.PoseLandmark.LEFT_HIP,
        'RIGHT_HIP': mp_pose.PoseLandmark.RIGHT_HIP,
        'LEFT_KNEE': mp_pose.PoseLandmark.LEFT_KNEE,
        'RIGHT_KNEE': mp_pose.PoseLandmark.RIGHT_KNEE,
        'LEFT_ANKLE': mp_pose.PoseLandmark.LEFT_ANKLE,
        'RIGHT_ANKLE': mp_pose.PoseLandmark.RIGHT_ANKLE
    }
    
    print("\n=== 主要キーポイントの座標 ===")
    for name, landmark_id in key_points.items():
        landmark = landmarks[landmark_id]
        x, y = int(landmark.x * w), int(landmark.y * h)
        confidence = landmark.visibility
        print(f"{name}: ({x}, {y}) - 信頼度: {confidence:.3f}")

def get_pose_landmarks_array(image_path):
    """
    骨格のランドマーク座標を配列として取得
    
    Returns:
        numpy.ndarray: shape=(33, 4) [x, y, z, visibility]
    """
    image = cv2.imread(image_path)
    if image is None:
        return None
    
    with mp_pose.Pose(
        static_image_mode=True,
        model_complexity=2,
        min_detection_confidence=0.5
    ) as pose:
        
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb_image)
        
        if results.pose_landmarks:
            landmarks = []
            for landmark in results.pose_landmarks.landmark:
                landmarks.append([
                    landmark.x, 
                    landmark.y, 
                    landmark.z, 
                    landmark.visibility
                ])
            return np.array(landmarks)
        else:
            return None

def batch_process_images(image_folder, output_folder):
    """
    フォルダ内の複数画像を一括処理
    """
    import os
    from pathlib import Path
    
    input_path = Path(image_folder)
    output_path = Path(output_folder)
    output_path.mkdir(exist_ok=True)
    
    # 対応する画像ファイル形式
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
    
    for img_file in input_path.iterdir():
        if img_file.suffix.lower() in image_extensions:
            print(f"処理中: {img_file.name}")
            output_file = output_path / f"pose_{img_file.name}"
            detect_pose_from_image(str(img_file), str(output_file))

if __name__ == "__main__":
    print("MediaPipe 静止画骨格抽出プログラム")
    print("=" * 40)
    
    # 使用例1: 単一画像の処理
    image_path = input("画像ファイルのパスを入力してください: ")
    
    if image_path.strip():
        # 結果画像の保存パスを生成
        base_name = image_path.rsplit('.', 1)[0]
        output_path = f"{base_name}_pose_result.jpg"
        
        # 骨格検出の実行
        results = detect_pose_from_image(image_path, output_path)
        
        if results:
            # ランドマーク座標配列の取得
            landmarks_array = get_pose_landmarks_array(image_path)
            if landmarks_array is not None:
                print(f"\n=== ランドマーク配列情報 ===")
                print(f"配列サイズ: {landmarks_array.shape}")
                print(f"検出されたキーポイント数: {len(landmarks_array)}")
                
                # 配列の保存（オプション）
                np.save(f"{base_name}_landmarks.npy", landmarks_array)
                print(f"ランドマーク配列を保存しました: {base_name}_landmarks.npy")
    else:
        print("画像パスが入力されませんでした")