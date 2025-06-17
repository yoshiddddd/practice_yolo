import cv2
import mediapipe as mp
import numpy as np

# MediaPipe pose初期化
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Poseランドマークのインデックス（33個のキーポイント）
# 顔部分のランドマーク
NOSE = mp_pose.PoseLandmark.NOSE.value           # 0: 鼻
LEFT_EYE_INNER = mp_pose.PoseLandmark.LEFT_EYE_INNER.value    # 1: 左目内側
LEFT_EYE = mp_pose.PoseLandmark.LEFT_EYE.value               # 2: 左目
LEFT_EYE_OUTER = mp_pose.PoseLandmark.LEFT_EYE_OUTER.value    # 3: 左目外側
RIGHT_EYE_INNER = mp_pose.PoseLandmark.RIGHT_EYE_INNER.value  # 4: 右目内側
RIGHT_EYE = mp_pose.PoseLandmark.RIGHT_EYE.value             # 5: 右目
RIGHT_EYE_OUTER = mp_pose.PoseLandmark.RIGHT_EYE_OUTER.value  # 6: 右目外側
LEFT_EAR = mp_pose.PoseLandmark.LEFT_EAR.value               # 7: 左耳
RIGHT_EAR = mp_pose.PoseLandmark.RIGHT_EAR.value             # 8: 右耳

def detect_pose_ear_nose_and_draw_lines(video_path=0, output_path=None):
    """
    骨格抽出から耳と鼻を検出し、線を描画する
    
    Args:
        video_path: 動画ファイルパス（0でウェブカメラ）
        output_path: 出力動画パス（Noneの場合はリアルタイム表示のみ）
    """
    
    # 動画キャプチャ初期化
    cap = cv2.VideoCapture(video_path)
    
    # 出力動画の設定
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Pose初期化
    with mp_pose.Pose(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        model_complexity=2  # 高精度モード（0: Lite, 1: Full, 2: Heavy）
    ) as pose:
        
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                print("動画の読み込みが完了しました。")
                break
            
            # BGR画像をRGBに変換
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image_rgb.flags.writeable = False
            
            # 骨格検出実行
            results = pose.process(image_rgb)
            
            # 描画用に画像を再度書き込み可能にする
            image_rgb.flags.writeable = True
            image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
            
            # 骨格ランドマークが検出された場合
            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark
                h, w, _ = image.shape
                
                # 各ランドマークの可視性チェック関数
                def is_visible(landmark):
                    return landmark.visibility > 0.5
                
                # 座標取得関数
                def get_coordinates(landmark_idx):
                    landmark = landmarks[landmark_idx]
                    if is_visible(landmark):
                        x = int(landmark.x * w)
                        y = int(landmark.y * h)
                        return (x, y), landmark.visibility
                    return None, 0
                
                # 各部位の座標取得
                nose_coords, nose_vis = get_coordinates(NOSE)
                left_ear_coords, left_ear_vis = get_coordinates(LEFT_EAR)
                right_ear_coords, right_ear_vis = get_coordinates(RIGHT_EAR)
                
                # 検出された部位を描画
                points_detected = []
                
                if nose_coords:
                    cv2.circle(image, nose_coords, 8, (0, 255, 0), -1)  # 緑色の鼻
                    cv2.putText(image, f'Nose({nose_vis:.2f})', 
                               (nose_coords[0] + 10, nose_coords[1]), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    points_detected.append(('nose', nose_coords))
                
                if left_ear_coords:
                    cv2.circle(image, left_ear_coords, 8, (255, 0, 0), -1)  # 青色の左耳
                    cv2.putText(image, f'L.Ear({left_ear_vis:.2f})', 
                               (left_ear_coords[0] + 10, left_ear_coords[1]), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                    points_detected.append(('left_ear', left_ear_coords))
                
                if right_ear_coords:
                    cv2.circle(image, right_ear_coords, 8, (255, 0, 0), -1)  # 青色の右耳
                    cv2.putText(image, f'R.Ear({right_ear_vis:.2f})', 
                               (right_ear_coords[0] + 10, right_ear_coords[1]), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                    points_detected.append(('right_ear', right_ear_coords))
                
                # 線を描画
                # 鼻から両耳への線
                if nose_coords and left_ear_coords:
                    cv2.line(image, nose_coords, left_ear_coords, (0, 255, 255), 3)  # 黄色の線
                
                if nose_coords and right_ear_coords:
                    cv2.line(image, nose_coords, right_ear_coords, (0, 255, 255), 3)  # 黄色の線
                
                # 両耳を繋ぐ線
                if left_ear_coords and right_ear_coords:
                    cv2.line(image, left_ear_coords, right_ear_coords, (255, 255, 0), 3)  # シアン色の線
                
                # 三角形を描画（すべての点が検出された場合）
                if nose_coords and left_ear_coords and right_ear_coords:
                    # 三角形の輪郭を描画
                    triangle_points = np.array([nose_coords, left_ear_coords, right_ear_coords], np.int32)
                    triangle_points = triangle_points.reshape((-1, 1, 2))
                    cv2.polylines(image, [triangle_points], True, (255, 0, 255), 2)  # マゼンタ色の三角形
                    
                    # 三角形の面積計算と表示
                    area = cv2.contourArea(triangle_points)
                    cv2.putText(image, f'Triangle Area: {area:.0f}px²', 
                               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
                
                # 全体の骨格を薄く描画
                mp_drawing.draw_landmarks(
                    image,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
                
                # 検出情報を画面に表示
                detection_info = f"Detected: {len(points_detected)}/3 points"
                cv2.putText(image, detection_info, (10, h - 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
            else:
                # 骨格が検出されない場合
                cv2.putText(image, "No pose detected", (10, 50), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            # FPS表示
            fps_text = f"Press 'q' to quit"
            cv2.putText(image, fps_text, (10, h - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # 出力動画に書き込み
            if output_path:
                out.write(image)
            
            # リアルタイム表示
            cv2.imshow('Pose-based Ear-Nose Detection', image)
            
            # 'q'キーで終了
            if cv2.waitKey(5) & 0xFF == ord('q'):
                break
    
    # リソース解放
    cap.release()
    if output_path:
        out.release()
    cv2.destroyAllWindows()

def show_pose_landmarks_info():
    """Poseランドマークの情報を表示"""
    print("\n=== MediaPipe Pose ランドマーク情報 ===")
    print(f"0: NOSE (鼻)")
    print(f"1: LEFT_EYE_INNER (左目内側)")
    print(f"2: LEFT_EYE (左目)")
    print(f"3: LEFT_EYE_OUTER (左目外側)")
    print(f"4: RIGHT_EYE_INNER (右目内側)")
    print(f"5: RIGHT_EYE (右目)")
    print(f"6: RIGHT_EYE_OUTER (右目外側)")
    print(f"7: LEFT_EAR (左耳)")
    print(f"8: RIGHT_EAR (右耳)")
    print(f"9: MOUTH_LEFT (口左)")
    print(f"10: MOUTH_RIGHT (口右)")
    print("...")
    print("※このプログラムでは鼻(0)、左耳(7)、右耳(8)を使用します")
    print("※可視性(visibility)が0.5以上の点のみを使用します\n")

def main():
    """メイン関数"""
    
    print("MediaPipe 骨格抽出による耳鼻検出プログラム")
    show_pose_landmarks_info()
    
    print("1. ウェブカメラを使用")
    print("2. 動画ファイルを使用")
    print("3. ランドマーク情報のみ表示")
    
    choice = input("選択してください (1, 2, or 3): ")
    
    if choice == "1":
        # ウェブカメラ使用
        print("ウェブカメラを起動します...")
        print("'q'キーで終了")
        detect_pose_ear_nose_and_draw_lines(video_path=0)
        
    elif choice == "2":
        # 動画ファイル使用
        video_path = input("動画ファイルのパスを入力してください: ")
        save_output = input("結果を動画ファイルに保存しますか？ (y/n): ")
        
        if save_output.lower() == 'y':
            output_path = input("出力ファイル名を入力してください（例: output.mp4）: ")
            detect_pose_ear_nose_and_draw_lines(video_path, output_path)
            print(f"結果が {output_path} に保存されました")
        else:
            detect_pose_ear_nose_and_draw_lines(video_path)
            
    elif choice == "3":
        # ランドマーク情報のみ表示
        show_pose_landmarks_info()
    
    else:
        print("無効な選択です")

if __name__ == "__main__":
    main()