import cv2
import easyocr
import numpy as np
import re

class LicensePlateRecognitionCNN:
    def __init__(self, languages=["en"], gpu=False):
        self.reader = easyocr.Reader(languages, gpu=gpu)

    def detect_and_recognize(self, image_path, visualize=False):
        """
        Multi-strategy approach untuk license plate detection:
        1. Try contour detection
        2. Fallback: Apply OCR to whole image with smart filtering
        """
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Cannot read image: {image_path}")

        result_img = img.copy() if visualize else None

        # Case 1: Try Contour Detection
        plate_roi, plate_location = self._detect_plate_roi(img)
        
        if plate_roi is not None and plate_roi.size > 100:
            plate_text, confidence = self._recognize_text_from_roi(plate_roi)
            
            if visualize and plate_location is not None:
                cv2.polylines(result_img, [plate_location], True, (0, 255, 0), 3)
        else:
            # Case 2: Fallback - OCR Full Image with Smart Filtering
            plate_text, confidence, detected_boxes = self._ocr_with_smart_filtering(img)
            
            if visualize and detected_boxes:
                for box in detected_boxes:
                    cv2.polylines(result_img, [box], True, (0, 255, 0), 2)
        
        return plate_text, confidence, result_img
    
    def _detect_plate_roi(self, img):
        """Deteksi ROI plat nomor menggunakan contour detection"""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        bilateral = cv2.bilateralFilter(gray, 11, 17, 17)
        edged = cv2.Canny(bilateral, 30, 200)
        
        contours, _ = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:30]
        
        plate_location = None
        
        for contour in contours:
            perimeter = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.018 * perimeter, True)
            
            # Check for rectangular shape
            if len(approx) == 4:
                x, y, w, h = cv2.boundingRect(approx)
                aspect_ratio = w / float(h)
                
                # Plat nomor criteria
                if 2.0 <= aspect_ratio <= 5.5 and h >= img.shape[0] * 0.04:
                    plate_location = approx
                    break
        
        if plate_location is not None:
            # Crop dengan margin
            x, y, w, h = cv2.boundingRect(plate_location)
            margin = 5
            x = max(0, x - margin)
            y = max(0, y - margin)
            w = min(img.shape[1] - x, w + 2*margin)
            h = min(img.shape[0] - y, h + 2*margin)
            
            cropped = img[y:y+h, x:x+w]
            return cropped, plate_location
        
        return None, None
    
    def _recognize_text_from_roi(self, plate_roi):
        """OCR pada cropped plate ROI - simple approach, let EasyOCR detect spacing naturally"""
        if plate_roi is None or plate_roi.size == 0:
            return "Tidak terdeteksi", 0.0
        
        # Preprocessing
        gray = cv2.cvtColor(plate_roi, cv2.COLOR_BGR2GRAY)
        
        # Multiple preprocessing attempts untuk improve accuracy
        attempts = []
        
        # Attempt 1: CLAHE + Otsu
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        _, thresh1 = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        attempts.append(('CLAHE+Otsu', thresh1))
        
        # Attempt 2: Adaptive threshold
        thresh2 = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                        cv2.THRESH_BINARY, 11, 2)
        attempts.append(('Adaptive', thresh2))
        
        # Attempt 3: High contrast CLAHE + Otsu
        clahe_high = cv2.createCLAHE(clipLimit=3.5, tileGridSize=(8, 8))
        enhanced_high = clahe_high.apply(gray)
        _, thresh3 = cv2.threshold(enhanced_high, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        attempts.append(('HighCLAHE+Otsu', thresh3))
        
        # Attempt 4: Bilateral filter + Otsu
        bilateral = cv2.bilateralFilter(gray, 9, 75, 75)
        _, thresh4 = cv2.threshold(bilateral, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        attempts.append(('Bilateral+Otsu', thresh4))
        
        # Attempt 5: Inverted (untuk plat with dark background)
        _, thresh5 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        attempts.append(('Inverted', thresh5))
        
        # Try OCR on each preprocessing
        best_result = ("", 0.0)
        
        for name, preprocessed in attempts:
            try:
                results = self.reader.readtext(
                    preprocessed,
                    detail=1,
                    paragraph=False,
                    allowlist='0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ '
                )
                
                if results:
                    sorted_results = sorted(results, key=lambda x: x[0][0][0])
                    
                    texts = []
                    confs = []
                    
                    for bbox, text, confidence in sorted_results:
                        cleaned = text.upper()
                        cleaned = re.sub(r"[^A-Z0-9 ]", "", cleaned)
                        cleaned = " ".join(cleaned.split())
                        
                        if cleaned.strip():
                            texts.append(cleaned)
                            confs.append(confidence)
                    
                    if texts:
                        plate_text = " ".join(texts)
                        avg_conf = float(np.mean(confs))
                        score = avg_conf * (1 + len(plate_text) * 0.05)
                        current_score = best_result[1] * (1 + len(best_result[0]) * 0.05) if best_result[0] else 0
                        
                        if score > current_score:
                            best_result = (plate_text, avg_conf)
            except Exception as e:
                continue
        
        return best_result if best_result[0] else ("Tidak terdeteksi", 0.0)
    
    
    
    def _ocr_with_smart_filtering(self, img):
        """
        Apply OCR ke seluruh gambar dengan smart filtering
        untuk mengambil hanya text yang kemungkinan plat nomor
        """
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Preprocessing
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        
        # OCR
        results = self.reader.readtext(
            enhanced,
            detail=1,
            paragraph=False,
            allowlist='0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'
        )
        
        if not results:
            return "Tidak terdeteksi", 0.0, []
        
        img_height, img_width = img.shape[:2]
        candidates = []
        
        for bbox, text, confidence in results:
            cleaned = re.sub(r"[^A-Z0-9]", "", text.upper())
            
            # Skip jika terlalu pendek
            if len(cleaned) < 4:
                continue
            
            # Hitung posisi dan ukuran
            x_coords = [pt[0] for pt in bbox]
            y_coords = [pt[1] for pt in bbox]
            
            x_min, x_max = min(x_coords), max(x_coords)
            y_min, y_max = min(y_coords), max(y_coords)
            
            width = x_max - x_min
            height = y_max - y_min
            
            # Position (lebih pilih gambar di bawah)
            y_center = (y_min + y_max) / 2
            y_ratio = y_center / img_height
            
            # Aspect ratio
            aspect_ratio = width / height if height > 0 else 0
            
            score = 0.0
            if y_ratio > 0.5:
                score += 2.0
            
            # Lebih pilih yang lebih lebar
            if 2.0 <= aspect_ratio <= 6.0:
                score += 2.0
            
            # Pilih yg lebih panjang (plat biasanya 6-10 karakter)
            if 6 <= len(cleaned) <= 12:
                score += 1.5
            elif 4 <= len(cleaned) <= 15:
                score += 1.0
            
            score += confidence
            
            size_ratio = (width * height) / (img_width * img_height)
            if 0.01 <= size_ratio <= 0.15:
                score += 1.0
            
            bbox_np = np.array(bbox, dtype=np.int32)
            
            candidates.append({
                'text': cleaned,
                'confidence': confidence,
                'score': score,
                'bbox': bbox_np
            })
        
        candidates.sort(key=lambda x: x['score'], reverse=True)
        
        if candidates:
            # Ambil candidate terbaik
            best = candidates[0]
            
            # Jika ada multiple candidates dengan score tinggi, gabungkan yang berdekatan
            if len(candidates) > 1:
                combined_text = best['text']
                combined_boxes = [best['bbox']]
                
                for candidate in candidates[1:3]:  # Cek top 3
                    if candidate['score'] > best['score'] * 0.7:
                        combined_text += candidate['text']
                        combined_boxes.append(candidate['bbox'])
                
                return combined_text, best['confidence'], combined_boxes
            
            return best['text'], best['confidence'], [best['bbox']]
        
        return "Tidak terdeteksi", 0.0, []