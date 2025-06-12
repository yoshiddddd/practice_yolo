import cv2
import numpy as np
from collections import defaultdict, deque
import json
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
import math

@dataclass
class StoreZone:
    """店舗内のエリア定義"""
    name: str
    polygon: List[Tuple[int, int]]  # エリアの境界座標
    color: Tuple[int, int, int]

@dataclass
class CustomerPath:
    """顧客の移動軌跡"""
    customer_id: int
    entry_time: datetime
    exit_time: Optional[datetime]
    path_points: List[Tuple[int, int, datetime]]
    visited_zones: List[str]
    reached_register: bool
    total_time: Optional[float]  # 滞在時間（秒）

class CustomerTracker:
    """顧客追跡・分析システム"""
    
    def __init__(self, video_path: str, store_zones: Dict[str, StoreZone]):
        self.video_path = video_path
        self.store_zones = store_zones
        
        # 追跡データ
        self.customer_paths: Dict[int, CustomerPath] = {}
        self.active_customers: Dict[int, deque] = defaultdict(lambda: deque(maxlen=50))
        
        # 分析結果
        self.non_purchasing_customers = []
        self.customer_stats = {}
        
        # YOLOv5の初期化（人物検出用）
        self.init_person_detector()
        
        # カメラキャリブレーション設定
        self.homography_matrix = None
        self.setup_camera_calibration()
    
    def init_person_detector(self):
        """人物検出モデルの初期化"""
        try:
            # YOLOv5を使用（pip install ultralytics）
            from ultralytics import YOLO
            self.detector = YOLO('yolov8n.pt')  # 軽量モデル
            print("YOLOv8 人物検出モデルを読み込みました")
        except ImportError:
            print("ultralyticsライブラリが見つかりません。")
            print("pip install ultralytics でインストールしてください")
            self.detector = None
    
    def setup_camera_calibration(self):
        """カメラキャリブレーション（実座標変換用）"""
        # 店舗の実際の座標点とカメラ画像上の対応点
        # 実際の運用では店舗レイアウトに合わせて調整が必要
        real_points = np.float32([
            [0, 0],      # 入口左
            [10, 0],     # 入口右  
            [10, 8],     # 店舗奥右
            [0, 8]       # 店舗奥左
        ])
        
        # カメラ画像上の対応点（ピクセル座標）
        image_points = np.float32([
            [100, 500],  # 入口左
            [500, 500],  # 入口右
            [450, 100],  # 店舗奥右
            [150, 100]   # 店舗奥左
        ])
        
        self.homography_matrix = cv2.getPerspectiveTransform(image_points, real_points)
    
    def detect_persons(self, frame):
        """フレームから人物を検出"""
        if self.detector is None:
            return []
        
        results = self.detector(frame, classes=[0], conf=0.5)  # class 0 = person
        detections = []
        
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = box.conf[0].cpu().numpy()
                    
                    # バウンディングボックスの中心点（足元付近）
                    center_x = int((x1 + x2) / 2)
                    center_y = int(y2)  # 足元の座標
                    
                    detections.append({
                        'bbox': (int(x1), int(y1), int(x2), int(y2)),
                        'center': (center_x, center_y),
                        'confidence': float(conf)
                    })
        
        return detections
    
    def simple_tracking(self, detections, frame_time):
        """シンプルな人物追跡（距離ベース）"""
        current_centers = [det['center'] for det in detections]
        
        # 新しい検出結果と既存の追跡対象をマッチング
        for i, center in enumerate(current_centers):
            best_match_id = None
            min_distance = float('inf')
            
            # 既存の追跡対象との距離を計算
            for customer_id, path_history in self.active_customers.items():
                if len(path_history) > 0:
                    last_pos = path_history[-1][0]  # 最後の位置
                    distance = math.sqrt((center[0] - last_pos[0])**2 + 
                                       (center[1] - last_pos[1])**2)
                    
                    if distance < min_distance and distance < 100:  # 閾値100ピクセル
                        min_distance = distance
                        best_match_id = customer_id
            
            # マッチした場合は既存IDを更新、そうでなければ新規作成
            if best_match_id is not None:
                self.active_customers[best_match_id].append((center, frame_time))
            else:
                # 新規顧客
                new_id = len(self.customer_paths) + 1
                self.active_customers[new_id].append((center, frame_time))
                
                # 新規顧客パス作成
                self.customer_paths[new_id] = CustomerPath(
                    customer_id=new_id,
                    entry_time=frame_time,
                    exit_time=None,
                    path_points=[(center[0], center[1], frame_time)],
                    visited_zones=[],
                    reached_register=False,
                    total_time=None
                )
    
    def point_in_polygon(self, point, polygon):
        """点がポリゴン内にあるかチェック"""
        x, y = point
        n = len(polygon)
        inside = False
        
        p1x, p1y = polygon[0]
        for i in range(1, n + 1):
            p2x, p2y = polygon[i % n]
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y
        
        return inside
    
    def update_zone_visits(self, customer_id, current_pos):
        """顧客の現在位置からゾーン訪問を更新"""
        if customer_id not in self.customer_paths:
            return
        
        customer = self.customer_paths[customer_id]
        
        for zone_name, zone in self.store_zones.items():
            if self.point_in_polygon(current_pos, zone.polygon):
                if zone_name not in customer.visited_zones:
                    customer.visited_zones.append(zone_name)
                
                # レジエリアに到達したかチェック
                if zone_name == "register":
                    customer.reached_register = True
    
    def cleanup_inactive_customers(self, current_time, timeout_seconds=30):
        """非アクティブな顧客を削除"""
        inactive_customers = []
        
        for customer_id, path_history in self.active_customers.items():
            if len(path_history) > 0:
                last_time = path_history[-1][1]
                if (current_time - last_time).total_seconds() > timeout_seconds:
                    inactive_customers.append(customer_id)
        
        # 非アクティブ顧客の退店処理
        for customer_id in inactive_customers:
            if customer_id in self.customer_paths:
                customer = self.customer_paths[customer_id]
                customer.exit_time = current_time
                customer.total_time = (customer.exit_time - customer.entry_time).total_seconds()
                
                # レジに到達していない顧客を記録
                if not customer.reached_register:
                    self.non_purchasing_customers.append(customer_id)
            
            del self.active_customers[customer_id]
    
    def process_video(self):
        """動画を処理して顧客追跡を実行"""
        cap = cv2.VideoCapture(self.video_path)
        frame_count = 0
        start_time = datetime.now()
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # フレーム時刻を計算
            frame_time = start_time + timedelta(seconds=frame_count * (1/30))  # 30FPS想定
            
            # 人物検出
            detections = self.detect_persons(frame)
            
            # 追跡更新
            self.simple_tracking(detections, frame_time)
            
            # ゾーン訪問更新
            for customer_id, path_history in self.active_customers.items():
                if len(path_history) > 0:
                    current_pos = path_history[-1][0]
                    self.update_zone_visits(customer_id, current_pos)
                    
                    # パス履歴更新
                    if customer_id in self.customer_paths:
                        self.customer_paths[customer_id].path_points.append(
                            (current_pos[0], current_pos[1], frame_time)
                        )
            
            # 非アクティブ顧客の清理
            self.cleanup_inactive_customers(frame_time)
            
            # 結果を描画
            annotated_frame = self.draw_tracking_results(frame)
            
            # 表示
            cv2.imshow('Customer Tracking', annotated_frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
            frame_count += 1
            
            # 進捗表示
            if frame_count % 100 == 0:
                print(f"処理済みフレーム: {frame_count}")
        
        cap.release()
        cv2.destroyAllWindows()
        
        # 最終的な分析結果を生成
        self.generate_analysis_report()
    
    def draw_tracking_results(self, frame):
        """追跡結果を描画"""
        annotated_frame = frame.copy()
        
        # ストアゾーンを描画
        for zone_name, zone in self.store_zones.items():
            pts = np.array(zone.polygon, np.int32)
            pts = pts.reshape((-1, 1, 2))
            cv2.polylines(annotated_frame, [pts], True, zone.color, 2)
            cv2.putText(annotated_frame, zone_name, zone.polygon[0], 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, zone.color, 2)
        
        # アクティブ顧客の軌跡を描画
        for customer_id, path_history in self.active_customers.items():
            if len(path_history) > 1:
                # 軌跡線を描画
                points = [pos[0] for pos in path_history]
                for i in range(len(points) - 1):
                    cv2.line(annotated_frame, points[i], points[i+1], (0, 255, 0), 2)
                
                # 現在位置を描画
                current_pos = points[-1]
                cv2.circle(annotated_frame, current_pos, 5, (0, 0, 255), -1)
                cv2.putText(annotated_frame, f"ID:{customer_id}", 
                           (current_pos[0] + 10, current_pos[1] - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # 統計情報を表示
        info_text = [
            f"Active Customers: {len(self.active_customers)}",
            f"Non-purchasing: {len(self.non_purchasing_customers)}",
            f"Total Tracked: {len(self.customer_paths)}"
        ]
        
        for i, text in enumerate(info_text):
            cv2.putText(annotated_frame, text, (10, 30 + i * 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return annotated_frame
    
    def generate_analysis_report(self):
        """分析レポートを生成"""
        print("\n" + "="*50)
        print("店舗内顧客動線分析レポート")
        print("="*50)
        
        total_customers = len(self.customer_paths)
        non_purchasing_count = len(self.non_purchasing_customers)
        purchasing_count = total_customers - non_purchasing_count
        
        print(f"総顧客数: {total_customers}")
        print(f"購入顧客数: {purchasing_count}")
        print(f"非購入顧客数: {non_purchasing_count}")
        
        if total_customers > 0:
            conversion_rate = (purchasing_count / total_customers) * 100
            print(f"コンバージョン率: {conversion_rate:.1f}%")
        
        print("\n=== 非購入顧客の詳細 ===")
        for customer_id in self.non_purchasing_customers:
            customer = self.customer_paths[customer_id]
            print(f"顧客ID {customer_id}:")
            print(f"  滞在時間: {customer.total_time:.1f}秒")
            print(f"  訪問エリア: {', '.join(customer.visited_zones)}")
            print(f"  軌跡ポイント数: {len(customer.path_points)}")
        
        # 結果をJSONファイルに保存
        self.save_results_to_file()
    
    def save_results_to_file(self):
        """結果をファイルに保存"""
        results = {
            'analysis_time': datetime.now().isoformat(),
            'total_customers': len(self.customer_paths),
            'non_purchasing_customers': len(self.non_purchasing_customers),
            'customer_details': {}
        }
        
        for customer_id, customer in self.customer_paths.items():
            results['customer_details'][customer_id] = {
                'entry_time': customer.entry_time.isoformat(),
                'exit_time': customer.exit_time.isoformat() if customer.exit_time else None,
                'total_time': customer.total_time,
                'visited_zones': customer.visited_zones,
                'reached_register': customer.reached_register,
                'path_points_count': len(customer.path_points)
            }
        
        with open('customer_analysis_results.json', 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        print(f"\n分析結果を保存しました: customer_analysis_results.json")

def main():
    """メイン実行関数"""
    
    # 店舗レイアウトの定義（実際の店舗に合わせて調整）
    store_zones = {
        "entrance": StoreZone("entrance", [(50, 400), (200, 400), (200, 500), (50, 500)], (0, 255, 0)),
        "product_area1": StoreZone("product_area1", [(200, 200), (400, 200), (400, 400), (200, 400)], (255, 0, 0)),
        "product_area2": StoreZone("product_area2", [(400, 200), (600, 200), (600, 400), (400, 400)], (255, 0, 0)),
        "register": StoreZone("register", [(500, 450), (650, 450), (650, 550), (500, 550)], (0, 0, 255))
    }
    
    # 動画ファイルパス（実際のファイルパスに変更）
    video_path = input("防犯カメラ動画ファイルのパスを入力してください: ")
    
    if not video_path.strip():
        print("サンプル用のテスト動画を使用します")
        video_path = "aaa.mp4"  # サンプル動画
    
    # 追跡システムの初期化と実行
    tracker = CustomerTracker(video_path, store_zones)
    
    print("顧客追跡を開始します...")
    print("'q'キーで終了")
    
    tracker.process_video()

if __name__ == "__main__":
    main()