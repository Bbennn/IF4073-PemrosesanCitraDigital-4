import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import cv2
import os
import threading
import numpy as np

# Import modules
from fruit import FruitRecognitionCNN
from license_plate_recognition_cnn import LicensePlateRecognitionCNN
from human_tracking_yolo import MultipleHumanTracker

class ImageProcessingGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Tugas 4 IF4073 - Image Processing")
        self.root.geometry("1000x700")
        
        # Variables
        self.current_image_path = None
        self.current_video_path = None
        
        # Default paths
        self.default_model_dir = "./train_results"
        self.default_image_dir = "./dataset/fruit/test"
        self.default_plate_dir = "./dataset/plates"
        self.default_video_dir = "./dataset/videos"
        
        # Models
        self.fruit_model = None
        self.plate_model = None
        self.tracker_model = None
        
        # Model loading status
        self.plate_model_loaded = False
        self.tracker_model_loaded = False
        
        self.setup_gui()
        
        # Auto-load models di background
        self.auto_load_models()
        
    def setup_gui(self):
        # Title
        title_frame = tk.Frame(self.root, bg='#2C3E50', height=60)
        title_frame.pack(fill='x')
        title_frame.pack_propagate(False)
        
        tk.Label(
            title_frame,
            text="Sistem Pengolahan Citra Digital - IF4073",
            font=('Arial', 18, 'bold'),
            bg='#2C3E50',
            fg='white'
        ).pack(pady=15)
        
        # Notebook (Tabs)
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Tab 1: Fruit Recognition
        self.fruit_tab = self.create_fruit_tab()
        self.notebook.add(self.fruit_tab, text='Pengenalan Buah')
        
        # Tab 2: License Plate Recognition
        self.plate_tab = self.create_plate_tab()
        self.notebook.add(self.plate_tab, text='Pengenalan Plat Nomor')
        
        # Tab 3: Video Analysis
        self.video_tab = self.create_video_tab()
        self.notebook.add(self.video_tab, text='Analisis Video')
        
        # Status bar
        self.status_bar = tk.Label(
            self.root,
            text="Initializing...",
            bd=1,
            relief=tk.SUNKEN,
            anchor=tk.W
        )
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
    
    def auto_load_models(self):
        """Load Plate dan YOLO model otomatis di background"""
        def load_all_models():
            # Load Plate Model
            try:
                self.update_status("Loading OCR model...")
                self.update_plate_model_status("Loading EasyOCR model...")
                
                self.plate_model = LicensePlateRecognitionCNN(
                    languages=['en'],
                    gpu=False
                )
                
                self.plate_model_loaded = True
                self.update_plate_model_status("Model Ready")
                self.update_plate_info_status("Ready to recognize", "green")
            except Exception as e:
                self.update_plate_model_status(f"Error loading model")
                self.update_plate_info_status(f"Error: {str(e)}", "red")
            
            # Load YOLO Model
            try:
                self.update_status("Loading YOLO model...")
                self.update_video_model_status("Loading YOLOv8 model...")
                
                self.tracker_model = MultipleHumanTracker(
                    model_name='yolov8n.pt',
                    confidence_threshold=0.5
                )
                
                self.tracker_model_loaded = True
                self.update_video_model_status("Model Ready")
                self.update_video_info_status("Ready to track", "green")
            except Exception as e:
                self.update_video_model_status(f"Error loading model")
                self.update_video_info_status(f"Error: {str(e)}", "red")
            
            # Update status bar
            if self.plate_model_loaded and self.tracker_model_loaded:
                self.update_status("Ready - All models loaded")
            else:
                self.update_status("Ready - Some models failed to load")
        
        # Run in background thread
        threading.Thread(target=load_all_models, daemon=True).start()
    
    def update_plate_model_status(self, text):
        """Update model status di plate tab"""
        self.plate_model_status.config(text=text)
        self.root.update()
    
    def update_plate_info_status(self, text, color="black"):
        """Update info status di plate tab"""
        self.plate_info_status.config(text=text, fg=color)
        self.root.update()
    
    def update_video_model_status(self, text):
        """Update model status di video tab"""
        self.video_model_status.config(text=text)
        self.root.update()
    
    def update_video_info_status(self, text, color="black"):
        """Update info status di video tab"""
        self.video_info_status.config(text=text, fg=color)
        self.root.update()
    
    # FRUIT RECOGNITION TAB
    def create_fruit_tab(self):
        tab = tk.Frame(self.notebook)
        
        # Left panel
        left_panel = tk.Frame(tab, width=250, bg='#ECF0F1')
        left_panel.pack(side='left', fill='y', padx=5, pady=5)
        left_panel.pack_propagate(False)
        
        tk.Label(
            left_panel,
            text="Pengenalan Buah",
            font=('Arial', 12, 'bold'),
            bg='#ECF0F1'
        ).pack(pady=10)
        
        # Model status
        tk.Label(left_panel, text="Model Status:", bg='#ECF0F1', font=('Arial', 9, 'bold')).pack(pady=(10,2))
        self.fruit_model_status = tk.Label(
            left_panel,
            text="⚪ Model not loaded",
            font=('Arial', 9),
            bg='#ECF0F1',
            fg='gray'
        )
        self.fruit_model_status.pack(pady=2)
        
        tk.Button(
            left_panel,
            text="Load Model (.h5)",
            command=self.load_fruit_model,
            width=20,
            bg='#3498DB',
            fg='white'
        ).pack(pady=10)
        
        tk.Button(
            left_panel,
            text="Pilih Gambar",
            command=lambda: self.load_image('fruit'),
            width=20,
            bg='#2ECC71',
            fg='white'
        ).pack(pady=5)
        
        tk.Button(
            left_panel,
            text="Kenali Buah",
            command=self.recognize_fruit,
            width=20,
            bg='#E67E22',
            fg='white'
        ).pack(pady=5)
        
        # Info status
        tk.Label(left_panel, text="Status:", bg='#ECF0F1', font=('Arial', 9, 'bold')).pack(pady=(20,2))
        self.fruit_info_status = tk.Label(
            left_panel,
            text="Waiting for input",
            font=('Arial', 9),
            bg='#ECF0F1',
            fg='gray',
            wraplength=220
        )
        self.fruit_info_status.pack(pady=2)
        
        # Result frame
        result_frame = tk.LabelFrame(left_panel, text="Hasil Prediksi", bg='#ECF0F1', font=('Arial', 10, 'bold'))
        result_frame.pack(pady=20, padx=10, fill='x')
        
        tk.Label(result_frame, text="Buah:", bg='#ECF0F1', font=('Arial', 9)).pack(pady=2)
        self.fruit_prediction_label = tk.Label(
            result_frame,
            text="-",
            font=('Arial', 16, 'bold'),
            bg='#ECF0F1',
            fg='#2C3E50'
        )
        self.fruit_prediction_label.pack(pady=5)
        
        tk.Label(result_frame, text="Confidence:", bg='#ECF0F1', font=('Arial', 9)).pack(pady=2)
        self.fruit_confidence_label = tk.Label(
            result_frame,
            text="-",
            font=('Arial', 14, 'bold'),
            bg='#ECF0F1',
            fg='#27AE60'
        )
        self.fruit_confidence_label.pack(pady=5)
        
        # Right panel
        right_panel = tk.Frame(tab, bg='white')
        right_panel.pack(side='right', fill='both', expand=True, padx=5, pady=5)
        
        self.fruit_canvas = tk.Canvas(right_panel, bg='#34495E')
        self.fruit_canvas.pack(fill='both', expand=True)
        
        return tab
    
    # PLATE RECOGNITION TAB
    def create_plate_tab(self):
        tab = tk.Frame(self.notebook)
        
        # Left panel
        left_panel = tk.Frame(tab, width=250, bg='#ECF0F1')
        left_panel.pack(side='left', fill='y', padx=5, pady=5)
        left_panel.pack_propagate(False)
        
        tk.Label(
            left_panel,
            text="Pengenalan Plat Nomor",
            font=('Arial', 12, 'bold'),
            bg='#ECF0F1'
        ).pack(pady=10)
        
        # Model status
        tk.Label(left_panel, text="Model Status:", bg='#ECF0F1', font=('Arial', 9, 'bold')).pack(pady=(10,2))
        self.plate_model_status = tk.Label(
            left_panel,
            text="⏳ Loading model...",
            font=('Arial', 9),
            bg='#ECF0F1',
            fg='orange'
        )
        self.plate_model_status.pack(pady=2)
        
        tk.Button(
            left_panel,
            text="Pilih Gambar Plat",
            command=lambda: self.load_image('plate'),
            width=20,
            bg='#2ECC71',
            fg='white'
        ).pack(pady=10)
        
        tk.Button(
            left_panel,
            text="Kenali Plat",
            command=self.recognize_plate,
            width=20,
            bg='#E67E22',
            fg='white'
        ).pack(pady=5)
        
        # Info status
        tk.Label(left_panel, text="Status:", bg='#ECF0F1', font=('Arial', 9, 'bold')).pack(pady=(20,2))
        self.plate_info_status = tk.Label(
            left_panel,
            text="Loading model...",
            font=('Arial', 9),
            bg='#ECF0F1',
            fg='orange',
            wraplength=220
        )
        self.plate_info_status.pack(pady=2)
        
        # Result frame
        result_frame = tk.LabelFrame(left_panel, text="Hasil Deteksi", bg='#ECF0F1', font=('Arial', 10, 'bold'))
        result_frame.pack(pady=20, padx=10, fill='x')
        
        tk.Label(result_frame, text="Plat Nomor:", bg='#ECF0F1', font=('Arial', 9)).pack(pady=2)
        self.plate_number_label = tk.Label(
            result_frame,
            text="-",
            font=('Arial', 16, 'bold'),
            bg='#ECF0F1',
            fg='#2C3E50'
        )
        self.plate_number_label.pack(pady=5)
        
        tk.Label(result_frame, text="Confidence:", bg='#ECF0F1', font=('Arial', 9)).pack(pady=2)
        self.plate_confidence_label = tk.Label(
            result_frame,
            text="-",
            font=('Arial', 14, 'bold'),
            bg='#ECF0F1',
            fg='#27AE60'
        )
        self.plate_confidence_label.pack(pady=5)
        
        # Right panel
        right_panel = tk.Frame(tab, bg='white')
        right_panel.pack(side='right', fill='both', expand=True, padx=5, pady=5)
        
        self.plate_canvas = tk.Canvas(right_panel, bg='#34495E')
        self.plate_canvas.pack(fill='both', expand=True)
        
        return tab
    
    # VIDEO ANALYSIS TAB
    def create_video_tab(self):
        tab = tk.Frame(self.notebook)
        
        # Left panel
        left_panel = tk.Frame(tab, width=250, bg='#ECF0F1')
        left_panel.pack(side='left', fill='y', padx=5, pady=5)
        left_panel.pack_propagate(False)
        
        tk.Label(
            left_panel,
            text="Analisis Video (YOLO)",
            font=('Arial', 12, 'bold'),
            bg='#ECF0F1'
        ).pack(pady=10)
        
        # Model status
        tk.Label(left_panel, text="Model Status:", bg='#ECF0F1', font=('Arial', 9, 'bold')).pack(pady=(10,2))
        self.video_model_status = tk.Label(
            left_panel,
            text="⏳ Loading model...",
            font=('Arial', 9),
            bg='#ECF0F1',
            fg='orange'
        )
        self.video_model_status.pack(pady=2)
        
        tk.Button(
            left_panel,
            text="Pilih Video",
            command=self.load_video,
            width=20,
            bg='#2ECC71',
            fg='white'
        ).pack(pady=10)
        
        tk.Button(
            left_panel,
            text="Mulai Tracking",
            command=self.start_tracking,
            width=20,
            bg='#E67E22',
            fg='white'
        ).pack(pady=5)
        
        # Info status
        tk.Label(left_panel, text="Status:", bg='#ECF0F1', font=('Arial', 9, 'bold')).pack(pady=(20,2))
        self.video_info_status = tk.Label(
            left_panel,
            text="Loading model...",
            font=('Arial', 9),
            bg='#ECF0F1',
            fg='orange',
            wraplength=220
        )
        self.video_info_status.pack(pady=2)
        
        # Result frame
        result_frame = tk.LabelFrame(left_panel, text="Hasil Tracking", bg='#ECF0F1', font=('Arial', 10, 'bold'))
        result_frame.pack(pady=20, padx=10, fill='x')
        
        tk.Label(result_frame, text="Total Frames:", bg='#ECF0F1', font=('Arial', 9)).pack(pady=2)
        self.video_frames_label = tk.Label(
            result_frame,
            text="-",
            font=('Arial', 12, 'bold'),
            bg='#ECF0F1',
            fg='#2C3E50'
        )
        self.video_frames_label.pack(pady=2)
        
        tk.Label(result_frame, text="Unique Persons:", bg='#ECF0F1', font=('Arial', 9)).pack(pady=2)
        self.video_persons_label = tk.Label(
            result_frame,
            text="-",
            font=('Arial', 12, 'bold'),
            bg='#ECF0F1',
            fg='#2C3E50'
        )
        self.video_persons_label.pack(pady=2)
        
        tk.Label(result_frame, text="Avg FPS:", bg='#ECF0F1', font=('Arial', 9)).pack(pady=2)
        self.video_fps_label = tk.Label(
            result_frame,
            text="-",
            font=('Arial', 12, 'bold'),
            bg='#ECF0F1',
            fg='#27AE60'
        )
        self.video_fps_label.pack(pady=2)
        
        # Right panel
        right_panel = tk.Frame(tab, bg='white')
        right_panel.pack(side='right', fill='both', expand=True, padx=5, pady=5)
        
        self.video_canvas = tk.Canvas(right_panel, bg='#34495E')
        self.video_canvas.pack(fill='both', expand=True)
        
        return tab
    
    # UTILITY FUNCTIONS
    def update_status(self, message):
        self.status_bar.config(text=message)
        self.root.update()
    
    def display_image(self, image_path, canvas):
        img = Image.open(image_path)
        
        # Resize untuk fit canvas
        canvas.update()
        canvas_width = canvas.winfo_width()
        canvas_height = canvas.winfo_height()
        
        if canvas_width > 1 and canvas_height > 1:
            img.thumbnail((canvas_width - 20, canvas_height - 20), Image.Resampling.LANCZOS)
        
        photo = ImageTk.PhotoImage(img)
        canvas.delete("all")
        canvas.create_image(canvas_width // 2, canvas_height // 2, image=photo)
        canvas.image = photo
    
    def display_cv2_image(self, cv2_img, canvas):
        """Display OpenCV image (BGR) on canvas"""
        # Convert BGR to RGB
        img_rgb = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img_rgb)
        
        # Resize untuk fit canvas
        canvas.update()
        canvas_width = canvas.winfo_width()
        canvas_height = canvas.winfo_height()
        
        if canvas_width > 1 and canvas_height > 1:
            img.thumbnail((canvas_width - 20, canvas_height - 20), Image.Resampling.LANCZOS)
        
        photo = ImageTk.PhotoImage(img)
        canvas.delete("all")
        canvas.create_image(canvas_width // 2, canvas_height // 2, image=photo)
        canvas.image = photo
    
    def display_video_frame(self, frame, canvas):
        """Display video frame on canvas"""
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb)
        
        # Resize untuk fit canvas
        canvas.update()
        canvas_width = canvas.winfo_width()
        canvas_height = canvas.winfo_height()
        
        if canvas_width > 1 and canvas_height > 1:
            img.thumbnail((canvas_width - 20, canvas_height - 20), Image.Resampling.LANCZOS)
        
        photo = ImageTk.PhotoImage(img)
        canvas.delete("all")
        canvas.create_image(canvas_width // 2, canvas_height // 2, image=photo)
        canvas.image = photo
        self.root.update()
    
    # FRUIT FUNCTIONS
    def load_fruit_model(self):
        model_path = filedialog.askopenfilename(
            title="Pilih Model Fruit Recognition",
            initialdir=self.default_model_dir,
            filetypes=[("H5 files", "*.h5"), ("All files", "*.*")]
        )
        
        if not model_path:
            return
        
        try:
            self.update_status("Loading fruit model...")
            self.fruit_model_status.config(text="⏳ Loading model...", fg="orange")
            self.fruit_info_status.config(text="Loading model...", fg="orange")
            
            self.fruit_model = FruitRecognitionCNN(img_size=(224, 224))
            self.fruit_model.load_model(model_path)
            self.fruit_model.class_names = ['apple', 'banana', 'mixed', 'orange']
            
            self.fruit_model_status.config(text="Model Ready", fg="green")
            self.fruit_info_status.config(text="Ready to recognize", fg="green")
            self.update_status("Fruit model ready")
            
        except Exception as e:
            self.fruit_model_status.config(text=f"Error", fg="red")
            self.fruit_info_status.config(text=f"Error: {str(e)}", fg="red")
            messagebox.showerror("Error", str(e))
    
    def load_image(self, img_type):
        # Tentukan initial dir berdasarkan type
        if img_type == 'plate':
            initial_dir = self.default_plate_dir
        else:
            initial_dir = self.default_image_dir
        
        file_path = filedialog.askopenfilename(
            title=f"Pilih Gambar",
            initialdir=initial_dir,
            filetypes=[("Image files", "*.jpg *.jpeg *.png"), ("All files", "*.*")]
        )
        
        if not file_path:
            return
        
        self.current_image_path = file_path
        
        if img_type == 'fruit':
            self.display_image(file_path, self.fruit_canvas)
            self.fruit_info_status.config(text=f"Image loaded: {os.path.basename(file_path)}", fg="blue")
        elif img_type == 'plate':
            self.display_image(file_path, self.plate_canvas)
            self.plate_info_status.config(text=f"Image loaded: {os.path.basename(file_path)}", fg="blue")
        
        self.update_status(f"Image loaded: {os.path.basename(file_path)}")
    
    def recognize_fruit(self):
        if not self.current_image_path:
            messagebox.showwarning("Warning", "Please load an image first!")
            return
        
        if not self.fruit_model:
            messagebox.showwarning("Warning", "Please load the model first!")
            return
        
        try:
            self.update_status("Recognizing fruit...")
            self.fruit_info_status.config(text="Processing...", fg="orange")
            
            predicted_class, confidence, _ = self.fruit_model.predict_image(self.current_image_path)
            
            # Update result labels
            self.fruit_prediction_label.config(text=predicted_class.upper())
            self.fruit_confidence_label.config(text=f"{confidence:.1%}")
            
            self.fruit_info_status.config(text="Recognition complete", fg="green")
            self.update_status(f"Result: {predicted_class} ({confidence:.2%})")
            
        except Exception as e:
            self.fruit_info_status.config(text=f"Error: {str(e)}", fg="red")
            messagebox.showerror("Error", str(e))
    
    # PLATE FUNCTIONS
    def recognize_plate(self):
        if not self.current_image_path:
            messagebox.showwarning("Warning", "Please load an image first!")
            return
        
        if not self.plate_model_loaded:
            messagebox.showwarning("Warning", "Model is still loading, please wait...")
            return
        
        try:
            self.update_status("Recognizing plate...")
            self.plate_info_status.config(text="Processing...", fg="orange")
            
            # Detect dengan visualisasi bounding box
            plate_text, confidence, result_img = self.plate_model.detect_and_recognize(
                self.current_image_path,
                visualize=True
            )
            
            # Update result labels
            self.plate_number_label.config(text=plate_text)
            self.plate_confidence_label.config(text=f"{confidence:.1%}")
            
            # Display image dengan bounding box hijau
            if result_img is not None:
                self.display_cv2_image(result_img, self.plate_canvas)
            
            self.plate_info_status.config(text="Recognition complete", fg="green")
            self.update_status(f"Result: {plate_text}")
            
        except Exception as e:
            self.plate_info_status.config(text=f"Error: {str(e)}", fg="red")
            messagebox.showerror("Error", str(e))
    
    # VIDEO FUNCTIONS
    def load_video(self):
        file_path = filedialog.askopenfilename(
            title="Pilih Video",
            initialdir=self.default_video_dir,
            filetypes=[("Video files", "*.mp4 *.avi *.mov"), ("All files", "*.*")]
        )
        
        if not file_path:
            return
        
        self.current_video_path = file_path
        self.video_info_status.config(text=f"Video loaded: {os.path.basename(file_path)}", fg="blue")
        self.update_status(f"Video loaded: {os.path.basename(file_path)}")
    
    def start_tracking(self):
        if not self.current_video_path:
            messagebox.showwarning("Warning", "Please load a video first!")
            return
        
        if not self.tracker_model_loaded:
            messagebox.showwarning("Warning", "Model is still loading, please wait...")
            return
        
        output_path = filedialog.asksaveasfilename(
            title="Save Output Video",
            initialdir=self.default_video_dir,
            defaultextension=".mp4",
            filetypes=[("MP4 files", "*.mp4")]
        )
        
        if not output_path:
            return
        
        def process_in_thread():
            try:
                self.update_status("Processing video...")
                self.video_info_status.config(text="Processing video...", fg="orange")
                
                # Open video untuk display
                cap = cv2.VideoCapture(self.current_video_path)
                fps = int(cap.get(cv2.CAP_PROP_FPS))
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                
                # Setup video writer
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
                
                # Tracking variables
                unique_ids = set()
                frame_count = 0
                
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    frame_count += 1
                    
                    # YOLO tracking
                    results = self.tracker_model.model.track(
                        frame,
                        persist=True,
                        conf=0.5,
                        classes=[0],
                        verbose=False
                    )
                    
                    # Annotate frame
                    annotated_frame = self.tracker_model.annotate_frame(frame, results, frame_count)
                    
                    # Track unique IDs
                    if results[0].boxes is not None and results[0].boxes.id is not None:
                        for track_id in results[0].boxes.id.cpu().numpy():
                            unique_ids.add(int(track_id))
                    
                    # Write frame
                    out.write(annotated_frame)
                    
                    # Display frame setiap 5 frames (untuk performance)
                    if frame_count % 5 == 0:
                        self.display_video_frame(annotated_frame, self.video_canvas)
                        progress = (frame_count / total_frames) * 100
                        self.video_info_status.config(
                            text=f"Processing: {progress:.1f}% ({frame_count}/{total_frames})",
                            fg="orange"
                        )
                
                cap.release()
                out.release()
                
                # Update result labels
                self.video_frames_label.config(text=str(frame_count))
                self.video_persons_label.config(text=str(len(unique_ids)))
                avg_fps = frame_count / (frame_count / fps) if fps > 0 else 0
                self.video_fps_label.config(text=f"{avg_fps:.2f}")
                
                self.video_info_status.config(text="Tracking complete", fg="green")
                self.update_status("Tracking complete")
                messagebox.showinfo("Success", f"Video saved to:\n{output_path}")
                
            except Exception as e:
                self.video_info_status.config(text=f"Error: {str(e)}", fg="red")
                self.update_status("Error during tracking")
                messagebox.showerror("Error", str(e))
        
        threading.Thread(target=process_in_thread, daemon=True).start()


def main():
    root = tk.Tk()
    app = ImageProcessingGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()