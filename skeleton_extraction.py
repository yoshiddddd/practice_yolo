# main.py

import cv2
import config  # 設定ファイルをインポート
from gaze_estimator import GazeEstimator # 視線推定クラスをインポート
import visualizer # 描画モジュールをインポート

def main(image_path, output_path):
    try:
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"画像ファイルが見つかりません: {image_path}")
    except (FileNotFoundError, Exception) as e:
        print(f"エラー: {e}")
        return

    annotated_image = image.copy()
    h, w, _ = image.shape

    # 2. 視線推定
    estimator = GazeEstimator(
        config.STATIC_IMAGE_MODE,
        config.MODEL_COMPLEXITY,
        config.MIN_DETECTION_CONFIDENCE
    )
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = estimator.process_image(image_rgb)

    # 3. 結果の描画
    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark
        
        # 視線パラメータを計算
        eye_pos, ear_pos, angle_rad = estimator.get_gaze_parameters(landmarks, (h, w))
        
        # 見ている棚を特定
        target_shelf_name, target_shelf_obj = estimator.find_target_shelf(
            eye_pos, angle_rad, config.SHELF_BOUNDARIES, (h, w),
            config.GAZE_PROJECTION_STEPS, config.SHELF_DETECTION_X_OFFSET
        )
        
        # 描画処理
        annotated_image = visualizer.draw_shelf_boundaries(
            annotated_image, config.SHELF_BOUNDARIES, target_shelf_obj, config.SHELF_DETECTION_X_OFFSET
        )
        annotated_image = visualizer.draw_gaze_info(annotated_image, eye_pos, ear_pos, angle_rad)
        text = f"推定ターゲット: {target_shelf_name}"
        annotated_image = visualizer.draw_japanese_text(
            annotated_image, text, (50, 50), config.FONT_PATH, config.FONT_SIZE, (255, 255, 255)
        )
    else:
        # 人物を検出できなかった場合
        text = "人物を検出できませんでした"
        annotated_image = visualizer.draw_japanese_text(
            annotated_image, text, (50, 50), config.FONT_PATH, config.FONT_SIZE, (0, 0, 255)
        )

    # 4. 保存と表示
    cv2.imwrite(output_path, annotated_image)
    print(f"結果を '{output_path}' として保存しました。")

    # ウィンドウサイズを調整して表示
    display_h, display_w = annotated_image.shape[:2]
    cv2.namedWindow('Gaze Estimation Result', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Gaze Estimation Result', display_w // 2, display_h // 2) # 表示サイズを半分に
    cv2.imshow('Gaze Estimation Result', annotated_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    estimator.close()

if __name__ == '__main__':
    IMAGE_FILE = 'input-img/test.jpeg' # 入力画像
    OUTPUT_FILE = 'output-img/gaze_estimation_result.jpg' # 出力画像
    main(IMAGE_FILE, OUTPUT_FILE)