# visualizer.py

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import math
def draw_japanese_text(image, text, position, font_path, font_size, color):
    """Pillowを使って画像に日本語を描画する関数"""
    try:
        img_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img_pil)
        font = ImageFont.truetype(font_path, font_size)
        
        # 縁取り
        shadow_color = (0, 0, 0)
        for offset in [(-2, -2), (-2, 2), (2, -2), (2, 2)]:
            draw.text((position[0] + offset[0], position[1] + offset[1]), text, font=font, fill=shadow_color)
            
        draw.text(position, text, font=font, fill=color)
        return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    except IOError:
        print(f"フォントファイルが見つかりません: {font_path}")
        cv2.putText(image, "Font file not found.", position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
        return image

def draw_gaze_info(image, eye_pos, ear_pos, angle_rad):
    """目・耳の位置と、視線の方向を示す線を描画する"""
    h, w, _ = image.shape
    
    # 目と耳の位置に円を描画
    cv2.circle(image, eye_pos, 10, (0, 0, 255), -1, cv2.LINE_AA) # 目: 赤
    cv2.circle(image, ear_pos, 10, (255, 0, 0), -1, cv2.LINE_AA) # 耳: 青

    # 視線を描画（画像の端まで延長）
    # 視線の向きに応じて描画の終点を画面の端に設定
    if math.cos(angle_rad) > 0:
        end_x = w
    else:
        end_x = 0
    end_y = eye_pos[1] + math.tan(angle_rad) * (end_x - eye_pos[0])

    cv2.line(image, eye_pos, (int(end_x), int(end_y)), (0, 255, 0), 3, cv2.LINE_AA) # 視線: 緑
    return image

def draw_shelf_boundaries(image, shelves, target_shelf, x_offset):
    """棚の境界線を描画し、ターゲットの棚をハイライトする"""
    h, w, _ = image.shape
    for shelf in shelves:
        # 棚の当たり判定エリアを描画
        cv2.rectangle(image, (w - x_offset, shelf["top_y"]), (w, shelf["bottom_y"]), (255, 255, 0), 2)
    
    # ターゲットになった棚を強調表示
    if target_shelf:
        cv2.rectangle(image, (w - x_offset, target_shelf["top_y"]), (w, target_shelf["bottom_y"]), (255, 0, 255), -1)
    
    return image