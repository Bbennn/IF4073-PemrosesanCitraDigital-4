import cv2
import numpy as np
from ultralytics import YOLO
import torch
from collections import defaultdict, deque
import time

class MultipleHumanTracker:
    def __init__(self, model_name='yolov8n.pt', confidence_threshold=0.5):
        """
        Inisialisasi tracker dengan YOLO model
        
        Args:
            model_name: nama model YOLO (default: 'yolov8n.pt')
            confidence_threshold: threshold untuk confidence detection
        """
        print("Memuat YOLO model...")
        self.model = YOLO(model_name)
        self.confidence_threshold = confidence_threshold
        
        # Dictionary untuk menyimpan track history
        self.track_history = defaultdict(lambda: deque(maxlen=30))
        
        # Colors untuk visualisasi berbeda untuk setiap ID
        self.colors = self.generate_colors(100)
        
        print(f"Model {model_name} berhasil dimuat!")
        
    def generate_colors(self, n):
        """
        Generate n warna berbeda untuk visualisasi
        
        Args:
            n: jumlah warna yang dibutuhkan
        
        Returns:
            list of BGR colors
        """
        colors = []
        for i in range(n):
            hue = int(180 * i / n)
            color = cv2.cvtColor(
                np.uint8([[[hue, 255, 255]]]), 
                cv2.COLOR_HSV2BGR
            )[0][0]
            colors.append(tuple(int(c) for c in color))
        return colors
    
    def process_video(self, video_path, output_path=None, show_live=True):
        """
        Proses video untuk tracking multiple humans
        
        Args:
            video_path: path ke file video input
            output_path: path untuk menyimpan video output (optional)
            show_live: tampilkan preview real-time (default: True)
        
        Returns:
            statistics: statistik tracking
        """
        # Buka video
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Tidak dapat membuka video: {video_path}")

        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"\nVideo Properties:")
        print(f"  Resolution: {width}x{height}")
        print(f"  FPS: {fps}")
        print(f"  Total Frames: {total_frames}")
        
        # Setup video writer jika output_path diberikan
        out = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        # Statistics
        stats = {
            'total_frames': 0,
            'total_detections': 0,
            'unique_ids': set(),
            'avg_persons_per_frame': 0,
            'processing_time': []
        }
        
        frame_count = 0
        
        print("\nMemulai tracking...")
        
        while True:
            ret, frame = cap.read()
            
            if not ret:
                break
            
            frame_count += 1
            start_time = time.time()
            
            # YOLO tracking (dengan built-in tracker)
            results = self.model.track(
                frame, 
                persist=True,
                conf=self.confidence_threshold,
                classes=[0], # kelas orang
                verbose=False
            )
            
            # Proses hasil tracking
            annotated_frame = self.annotate_frame(frame, results, frame_count)
            
            # Update statistics
            if results[0].boxes is not None and results[0].boxes.id is not None:
                num_detections = len(results[0].boxes.id)
                stats['total_detections'] += num_detections
                
                for track_id in results[0].boxes.id.cpu().numpy():
                    stats['unique_ids'].add(int(track_id))
            
            processing_time = time.time() - start_time
            stats['processing_time'].append(processing_time)
            
            # Tambahkan info processing
            info_text = [
                f"Frame: {frame_count}/{total_frames}",
                f"FPS: {1/processing_time:.1f}",
                f"Active Tracks: {len(results[0].boxes.id) if results[0].boxes.id is not None else 0}",
                f"Total Unique IDs: {len(stats['unique_ids'])}"
            ]
            
            y_offset = 30
            for text in info_text:
                cv2.putText(
                    annotated_frame,
                    text,
                    (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2
                )
                y_offset += 30
            
            # Simpan frame jika output_path diberikan
            if out:
                out.write(annotated_frame)
            
            # Tampilkan preview
            if show_live:
                cv2.imshow('Human Tracking', annotated_frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("\nTracking dihentikan oleh user")
                    break
            
            # Progress indicator
            if frame_count % 30 == 0:
                progress = (frame_count / total_frames) * 100
                print(f"Progress: {progress:.1f}% ({frame_count}/{total_frames} frames)")
        
        # Cleanup
        cap.release()
        if out:
            out.release()
        if show_live:
            cv2.destroyAllWindows()

        stats['total_frames'] = frame_count
        stats['avg_persons_per_frame'] = (
            stats['total_detections'] / frame_count if frame_count > 0 else 0
        )
        stats['avg_processing_time'] = np.mean(stats['processing_time'])
        stats['avg_fps'] = 1 / stats['avg_processing_time']
        
        print("\n" + "="*60)
        print("Tracking selesai!")
        print("="*60)
        self.print_statistics(stats)
        
        if output_path:
            print(f"\nVideo output disimpan ke: {output_path}")
        
        return stats
    
    def annotate_frame(self, frame, results, frame_num):
        """
        Anotasi frame dengan bounding boxes dan track IDs
        
        Args:
            frame: frame video
            results: hasil dari YOLO tracking
            frame_num: nomor frame
        
        Returns:
            annotated_frame
        """
        annotated_frame = frame.copy()
        
        # Cek apakah ada deteksi
        if results[0].boxes is None or results[0].boxes.id is None:
            return annotated_frame
        
        # Extract boxes dan IDs
        boxes = results[0].boxes.xyxy.cpu().numpy()
        confidences = results[0].boxes.conf.cpu().numpy()
        track_ids = results[0].boxes.id.cpu().numpy().astype(int)
        
        # Annotate setiap detection
        for box, conf, track_id in zip(boxes, confidences, track_ids):
            x1, y1, x2, y2 = map(int, box)
            
            color = self.colors[track_id % len(self.colors)]
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
            
            label = f"ID:{track_id} ({conf:.2f})"
            label_size, _ = cv2.getTextSize(
                label, 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.6, 
                2
            )
            
            cv2.rectangle(
                annotated_frame,
                (x1, y1 - label_size[1] - 10),
                (x1 + label_size[0], y1),
                color,
                -1
            )
            
            cv2.putText(
                annotated_frame,
                label,
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )
            
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            self.track_history[track_id].append((center_x, center_y))
            
            points = list(self.track_history[track_id])
            for i in range(1, len(points)):
                if points[i - 1] is None or points[i] is None:
                    continue
                
                thickness = int(np.sqrt(30 / float(i + 1)) * 2)
                cv2.line(
                    annotated_frame,
                    points[i - 1],
                    points[i],
                    color,
                    thickness
                )
        
        return annotated_frame
    
    def print_statistics(self, stats):
        """
        Print statistik tracking
        
        Args:
            stats: dictionary berisi statistik
        """
        print(f"\nStatistik Tracking:")
        print(f"  Total Frames Processed: {stats['total_frames']}")
        print(f"  Total Detections: {stats['total_detections']}")
        print(f"  Unique Persons Tracked: {len(stats['unique_ids'])}")
        print(f"  Average Persons per Frame: {stats['avg_persons_per_frame']:.2f}")
        print(f"  Average Processing Time: {stats['avg_processing_time']:.4f}s")
        print(f"  Average FPS: {stats['avg_fps']:.2f}")
    
    def process_youtube_video(self, youtube_url, output_path=None, 
                             max_duration=None):
        """
        Download dan proses video dari YouTube
        
        Args:
            youtube_url: URL video YouTube
            output_path: path untuk output video
            max_duration: durasi maksimal video dalam detik (optional)
        
        Returns:
            statistics
        """
        try:
            import yt_dlp
        except ImportError:
            print("yt-dlp tidak terinstal. Install dengan: pip install yt-dlp")
            return None
        
        print(f"Mengunduh video dari: {youtube_url}")
        
        ydl_opts = {
            'format': 'best[height<=720]',
            'outtmpl': '/tmp/downloaded_video.%(ext)s',
        }
        
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(youtube_url, download=True)
            video_path = f"/tmp/downloaded_video.{info['ext']}"
        
        print(f"Video berhasil diunduh: {video_path}")
        
        stats = self.process_video(video_path, output_path, show_live=False)
        
        return stats
    
    def create_summary_video(self, video_path, output_path, sample_rate=5):
        """
        Buat summary video dengan mengambil sample frames
        
        Args:
            video_path: path ke video input
            output_path: path untuk output summary
            sample_rate: ambil 1 frame setiap n frames
        """
        cap = cv2.VideoCapture(video_path)
        
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps//sample_rate, (width, height))
        
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_count % sample_rate == 0:
                # Process frame
                results = self.model.track(
                    frame, 
                    persist=True,
                    conf=self.confidence_threshold,
                    classes=[0],
                    verbose=False
                )
                
                annotated_frame = self.annotate_frame(frame, results, frame_count)
                out.write(annotated_frame)
            
            frame_count += 1
        
        cap.release()
        out.release()
        
        print(f"Summary video disimpan ke: {output_path}")