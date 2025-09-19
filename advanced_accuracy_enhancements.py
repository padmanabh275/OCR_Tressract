"""
Advanced Accuracy Enhancements
Additional techniques to maximize document extraction accuracy
"""

import cv2
import numpy as np
import pytesseract
import re
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import json
from pathlib import Path
import math
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
import warnings
warnings.filterwarnings('ignore')

# Optional imports with fallbacks
try:
    import imutils
    IMUTILS_AVAILABLE = True
except ImportError:
    IMUTILS_AVAILABLE = False
    print("⚠️ imutils not available, some features will be disabled")

try:
    from scipy import ndimage
    from skimage import restoration, segmentation, measure
    from skimage.filters import threshold_otsu, gaussian
    from skimage.morphology import disk, opening, closing
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("⚠️ scipy/skimage not available, some features will be disabled")

@dataclass
class AccuracyEnhancementResult:
    """Result of accuracy enhancement"""
    enhanced_image: np.ndarray
    confidence_boost: float
    enhancement_applied: List[str]
    processing_time: float

class AdvancedAccuracyEnhancements:
    """Advanced accuracy enhancement techniques"""
    
    def __init__(self):
        self.setup_enhancement_configs()
        
    def setup_enhancement_configs(self):
        """Setup enhancement configurations for different document types"""
        
        self.enhancement_configs = {
            'indian_documents': {
                'preprocessing': [
                    'denoise_advanced',
                    'contrast_enhancement',
                    'perspective_correction',
                    'rotation_correction',
                    'text_sharpening'
                ],
                'ocr_configs': [
                    '--psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-/:.,() ',
                    '--psm 4 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-/:.,() ',
                    '--psm 8 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-/:.,() ',
                    '--psm 13 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-/:.,() '
                ],
                'postprocessing': [
                    'text_cleaning',
                    'format_validation',
                    'confidence_boosting'
                ]
            },
            'international_documents': {
                'preprocessing': [
                    'denoise_advanced',
                    'contrast_enhancement',
                    'perspective_correction',
                    'rotation_correction',
                    'text_sharpening'
                ],
                'ocr_configs': [
                    '--psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-/:.,() ',
                    '--psm 4 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-/:.,() ',
                    '--psm 8 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-/:.,() '
                ],
                'postprocessing': [
                    'text_cleaning',
                    'format_validation',
                    'confidence_boosting'
                ]
            }
        }
    
    def enhance_document_accuracy(self, image: np.ndarray, document_type: str = 'unknown') -> AccuracyEnhancementResult:
        """Apply comprehensive accuracy enhancements"""
        
        import time
        start_time = time.time()
        
        enhanced_image = image.copy()
        enhancements_applied = []
        confidence_boost = 0.0
        
        # Determine enhancement config
        if 'indian' in document_type.lower() or document_type in ['pan_card', 'aadhaar_card', 'driving_license', 'voter_id', 'passport']:
            config = self.enhancement_configs['indian_documents']
        else:
            config = self.enhancement_configs['international_documents']
        
        # Apply preprocessing enhancements
        for enhancement in config['preprocessing']:
            try:
                if enhancement == 'denoise_advanced':
                    enhanced_image = self.advanced_denoising(enhanced_image)
                    enhancements_applied.append('advanced_denoising')
                    confidence_boost += 0.05
                
                elif enhancement == 'contrast_enhancement':
                    enhanced_image = self.advanced_contrast_enhancement(enhanced_image)
                    enhancements_applied.append('contrast_enhancement')
                    confidence_boost += 0.08
                
                elif enhancement == 'perspective_correction':
                    enhanced_image = self.perspective_correction(enhanced_image)
                    enhancements_applied.append('perspective_correction')
                    confidence_boost += 0.06
                
                elif enhancement == 'rotation_correction':
                    enhanced_image = self.rotation_correction(enhanced_image)
                    enhancements_applied.append('rotation_correction')
                    confidence_boost += 0.04
                
                elif enhancement == 'text_sharpening':
                    enhanced_image = self.advanced_text_sharpening(enhanced_image)
                    enhancements_applied.append('text_sharpening')
                    confidence_boost += 0.03
                
            except Exception as e:
                print(f"Enhancement {enhancement} failed: {e}")
                continue
        
        processing_time = time.time() - start_time
        
        return AccuracyEnhancementResult(
            enhanced_image=enhanced_image,
            confidence_boost=min(confidence_boost, 0.3),  # Cap at 30% boost
            enhancement_applied=enhancements_applied,
            processing_time=processing_time
        )
    
    def advanced_denoising(self, image: np.ndarray) -> np.ndarray:
        """Advanced denoising using multiple techniques"""
        try:
            # Convert to grayscale if needed
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image.copy()
            
            # Apply bilateral filter for edge-preserving smoothing
            denoised = cv2.bilateralFilter(gray, 9, 75, 75)
            
            # Apply non-local means denoising
            denoised = cv2.fastNlMeansDenoising(denoised, None, 10, 7, 21)
            
            # Apply median filter to remove salt and pepper noise
            denoised = cv2.medianBlur(denoised, 3)
            
            return denoised
            
        except Exception as e:
            print(f"Advanced denoising error: {e}")
            return image
    
    def advanced_contrast_enhancement(self, image: np.ndarray) -> np.ndarray:
        """Advanced contrast enhancement using multiple techniques"""
        try:
            # Convert to grayscale if needed
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image.copy()
            
            # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
            enhanced = clahe.apply(gray)
            
            # Apply gamma correction
            gamma = 1.2
            enhanced = np.power(enhanced / 255.0, gamma) * 255.0
            enhanced = np.uint8(enhanced)
            
            # Apply histogram equalization
            enhanced = cv2.equalizeHist(enhanced)
            
            return enhanced
            
        except Exception as e:
            print(f"Contrast enhancement error: {e}")
            return image
    
    def perspective_correction(self, image: np.ndarray) -> np.ndarray:
        """Correct perspective distortion"""
        try:
            # Convert to grayscale
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image.copy()
            
            # Detect edges
            edges = cv2.Canny(gray, 50, 150)
            
            # Find contours
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Find the largest contour (likely the document)
            if contours:
                largest_contour = max(contours, key=cv2.contourArea)
                
                # Approximate the contour
                epsilon = 0.02 * cv2.arcLength(largest_contour, True)
                approx = cv2.approxPolyDP(largest_contour, epsilon, True)
                
                # If we have 4 points, apply perspective correction
                if len(approx) == 4:
                    # Order points: top-left, top-right, bottom-right, bottom-left
                    points = approx.reshape(4, 2)
                    ordered_points = self.order_points(points)
                    
                    # Calculate dimensions
                    width = max(np.linalg.norm(ordered_points[0] - ordered_points[1]),
                               np.linalg.norm(ordered_points[2] - ordered_points[3]))
                    height = max(np.linalg.norm(ordered_points[0] - ordered_points[3]),
                                np.linalg.norm(ordered_points[1] - ordered_points[2]))
                    
                    # Define destination points
                    dst = np.array([
                        [0, 0],
                        [width - 1, 0],
                        [width - 1, height - 1],
                        [0, height - 1]
                    ], dtype=np.float32)
                    
                    # Calculate perspective transform matrix
                    matrix = cv2.getPerspectiveTransform(ordered_points.astype(np.float32), dst)
                    
                    # Apply perspective correction
                    corrected = cv2.warpPerspective(image, matrix, (int(width), int(height)))
                    return corrected
            
            return image
            
        except Exception as e:
            print(f"Perspective correction error: {e}")
            return image
    
    def order_points(self, pts):
        """Order points for perspective correction"""
        rect = np.zeros((4, 2), dtype=np.float32)
        
        # Sum and difference
        s = pts.sum(axis=1)
        diff = np.diff(pts, axis=1)
        
        # Top-left point has smallest sum
        rect[0] = pts[np.argmin(s)]
        
        # Bottom-right point has largest sum
        rect[2] = pts[np.argmax(s)]
        
        # Top-right point has smallest difference
        rect[1] = pts[np.argmin(diff)]
        
        # Bottom-left point has largest difference
        rect[3] = pts[np.argmax(diff)]
        
        return rect
    
    def rotation_correction(self, image: np.ndarray) -> np.ndarray:
        """Correct rotation using text orientation detection"""
        try:
            # Convert to grayscale
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image.copy()
            
            # Detect text orientation using Tesseract
            try:
                osd = pytesseract.image_to_osd(gray, output_type=pytesseract.Output.DICT)
                angle = osd['rotate']
                
                if angle != 0:
                    # Rotate image to correct orientation
                    if IMUTILS_AVAILABLE:
                        corrected = imutils.rotate_bound(image, -angle)
                    else:
                        corrected = self.rotate_image(image, -angle)
                    return corrected
            except:
                # Fallback: try different rotation angles
                best_angle = 0
                best_score = 0
                
                for angle in [-90, -45, -30, -15, 15, 30, 45, 90]:
                    if IMUTILS_AVAILABLE:
                        rotated = imutils.rotate_bound(image, angle)
                    else:
                        rotated = self.rotate_image(image, angle)
                    
                    try:
                        text = pytesseract.image_to_string(rotated, config='--psm 6')
                        score = len(text.strip())
                        if score > best_score:
                            best_score = score
                            best_angle = angle
                    except:
                        continue
                
                if best_angle != 0:
                    if IMUTILS_AVAILABLE:
                        corrected = imutils.rotate_bound(image, best_angle)
                    else:
                        corrected = self.rotate_image(image, best_angle)
                    return corrected
            
            return image
            
        except Exception as e:
            print(f"Rotation correction error: {e}")
            return image
    
    def rotate_image(self, image: np.ndarray, angle: float) -> np.ndarray:
        """Rotate image by angle (fallback when imutils not available)"""
        try:
            height, width = image.shape[:2]
            center = (width // 2, height // 2)
            
            # Get rotation matrix
            rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
            
            # Calculate new dimensions
            cos_angle = abs(rotation_matrix[0, 0])
            sin_angle = abs(rotation_matrix[0, 1])
            new_width = int((height * sin_angle) + (width * cos_angle))
            new_height = int((height * cos_angle) + (width * sin_angle))
            
            # Adjust rotation matrix for translation
            rotation_matrix[0, 2] += (new_width / 2) - center[0]
            rotation_matrix[1, 2] += (new_height / 2) - center[1]
            
            # Perform rotation
            rotated = cv2.warpAffine(image, rotation_matrix, (new_width, new_height))
            return rotated
            
        except Exception as e:
            print(f"Image rotation error: {e}")
            return image
    
    def advanced_text_sharpening(self, image: np.ndarray) -> np.ndarray:
        """Advanced text sharpening using multiple techniques"""
        try:
            # Convert to grayscale
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image.copy()
            
            # Apply unsharp masking
            gaussian = cv2.GaussianBlur(gray, (0, 0), 2.0)
            sharpened = cv2.addWeighted(gray, 1.5, gaussian, -0.5, 0)
            
            # Apply Laplacian sharpening
            laplacian = cv2.Laplacian(sharpened, cv2.CV_64F)
            laplacian = np.uint8(np.absolute(laplacian))
            sharpened = cv2.add(sharpened, laplacian)
            
            # Apply morphological operations
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
            sharpened = cv2.morphologyEx(sharpened, cv2.MORPH_CLOSE, kernel)
            
            return sharpened
            
        except Exception as e:
            print(f"Text sharpening error: {e}")
            return image
    
    def ensemble_ocr(self, image: np.ndarray, configs: List[str]) -> Dict[str, str]:
        """Apply ensemble OCR with multiple configurations"""
        results = {}
        
        for i, config in enumerate(configs):
            try:
                text = pytesseract.image_to_string(image, config=config)
                results[f'config_{i}'] = text.strip()
            except Exception as e:
                print(f"OCR config {i} failed: {e}")
                results[f'config_{i}'] = ""
        
        return results
    
    def text_cleaning(self, text: str) -> str:
        """Clean and normalize extracted text"""
        if not text:
            return ""
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove non-printable characters
        text = re.sub(r'[^\x20-\x7E]', '', text)
        
        # Fix common OCR errors
        text = text.replace('|', 'I')
        text = text.replace('0', 'O')
        text = text.replace('5', 'S')
        
        return text.strip()
    
    def format_validation(self, text: str, document_type: str) -> Dict[str, Any]:
        """Validate extracted text format"""
        validation = {
            'is_valid': True,
            'warnings': [],
            'errors': [],
            'format_score': 0.0
        }
        
        if document_type == 'pan_card':
            # Validate PAN format
            pan_match = re.search(r'[A-Z]{5}[0-9]{4}[A-Z]{1}', text)
            if pan_match:
                validation['format_score'] = 1.0
            else:
                validation['warnings'].append("PAN format not found")
                validation['format_score'] = 0.0
        
        elif document_type == 'aadhaar_card':
            # Validate Aadhaar format
            aadhaar_match = re.search(r'\d{4}\s?\d{4}\s?\d{4}', text)
            if aadhaar_match:
                validation['format_score'] = 1.0
            else:
                validation['warnings'].append("Aadhaar format not found")
                validation['format_score'] = 0.0
        
        return validation
    
    def confidence_boosting(self, text: str, document_type: str) -> float:
        """Calculate confidence boost based on text quality"""
        if not text:
            return 0.0
        
        boost = 0.0
        
        # Length factor
        if len(text) > 50:
            boost += 0.1
        
        # Character diversity
        unique_chars = len(set(text.lower()))
        if unique_chars > 20:
            boost += 0.05
        
        # Document-specific patterns
        if document_type == 'pan_card':
            if re.search(r'[A-Z]{5}[0-9]{4}[A-Z]{1}', text):
                boost += 0.15
            if 'INCOME TAX' in text.upper():
                boost += 0.1
        
        elif document_type == 'aadhaar_card':
            if re.search(r'\d{4}\s?\d{4}\s?\d{4}', text):
                boost += 0.15
            if 'AADHAAR' in text.upper():
                boost += 0.1
        
        return min(boost, 0.3)  # Cap at 30% boost

class MultiModelEnsemble:
    """Ensemble of multiple OCR models for maximum accuracy"""
    
    def __init__(self):
        self.models = []
        self.setup_models()
    
    def setup_models(self):
        """Setup multiple OCR models"""
        self.models = [
            {'name': 'tesseract_psm6', 'config': '--psm 6'},
            {'name': 'tesseract_psm4', 'config': '--psm 4'},
            {'name': 'tesseract_psm8', 'config': '--psm 8'},
            {'name': 'tesseract_psm13', 'config': '--psm 13'},
            {'name': 'tesseract_whitelist', 'config': '--psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-/:.,() '}
        ]
    
    def ensemble_extract(self, image: np.ndarray) -> Dict[str, Any]:
        """Extract text using ensemble of models"""
        results = {}
        
        for model in self.models:
            try:
                text = pytesseract.image_to_string(image, config=model['config'])
                results[model['name']] = {
                    'text': text.strip(),
                    'confidence': self.calculate_text_confidence(text),
                    'length': len(text.strip())
                }
            except Exception as e:
                print(f"Model {model['name']} failed: {e}")
                results[model['name']] = {
                    'text': '',
                    'confidence': 0.0,
                    'length': 0
                }
        
        # Find best result
        best_model = max(results.items(), key=lambda x: x[1]['confidence'])
        
        return {
            'best_model': best_model[0],
            'best_text': best_model[1]['text'],
            'best_confidence': best_model[1]['confidence'],
            'all_results': results
        }
    
    def calculate_text_confidence(self, text: str) -> float:
        """Calculate confidence score for text"""
        if not text:
            return 0.0
        
        score = 0.0
        
        # Length factor
        score += min(len(text) / 100, 1.0) * 0.3
        
        # Character diversity
        unique_chars = len(set(text.lower()))
        score += min(unique_chars / 26, 1.0) * 0.2
        
        # Pattern matching
        patterns = [
            r'[A-Z]{5}[0-9]{4}[A-Z]{1}',  # PAN format
            r'\d{4}\s?\d{4}\s?\d{4}',     # Aadhaar format
            r'[A-Z]{3}[0-9]{7}',          # EPIC format
            r'[A-Z]{1}[0-9]{7}',          # Passport format
            r'\d{3}-\d{2}-\d{4}',         # SSN format
            r'[A-Za-z\s]+',               # Name pattern
            r'\d{2}[/-]\d{2}[/-]\d{4}'    # Date pattern
        ]
        
        pattern_matches = sum(1 for pattern in patterns if re.search(pattern, text))
        score += (pattern_matches / len(patterns)) * 0.5
        
        return min(score, 1.0)

class AdaptivePreprocessing:
    """Adaptive preprocessing based on document characteristics"""
    
    def __init__(self):
        self.setup_adaptive_configs()
    
    def setup_adaptive_configs(self):
        """Setup adaptive preprocessing configurations"""
        self.configs = {
            'low_contrast': {
                'contrast_factor': 2.0,
                'brightness_factor': 1.2,
                'gamma': 0.8
            },
            'high_noise': {
                'denoise_strength': 'high',
                'bilateral_filter': True,
                'median_filter': True
            },
            'skewed_text': {
                'rotation_correction': True,
                'perspective_correction': True
            },
            'small_text': {
                'upscale_factor': 2.0,
                'sharpening': 'aggressive'
            }
        }
    
    def analyze_document(self, image: np.ndarray) -> Dict[str, Any]:
        """Analyze document characteristics"""
        analysis = {
            'contrast_level': 'normal',
            'noise_level': 'normal',
            'text_size': 'normal',
            'skew_angle': 0,
            'recommended_config': 'default'
        }
        
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Analyze contrast
        contrast = gray.std()
        if contrast < 30:
            analysis['contrast_level'] = 'low'
        elif contrast > 80:
            analysis['contrast_level'] = 'high'
        
        # Analyze noise
        noise = self.estimate_noise(gray)
        if noise > 0.1:
            analysis['noise_level'] = 'high'
        
        # Analyze text size
        text_size = self.estimate_text_size(gray)
        if text_size < 20:
            analysis['text_size'] = 'small'
        elif text_size > 40:
            analysis['text_size'] = 'large'
        
        # Analyze skew
        skew_angle = self.estimate_skew(gray)
        analysis['skew_angle'] = skew_angle
        if abs(skew_angle) > 5:
            analysis['recommended_config'] = 'skewed_text'
        
        return analysis
    
    def estimate_noise(self, image: np.ndarray) -> float:
        """Estimate noise level in image"""
        try:
            # Use Laplacian to estimate noise
            laplacian = cv2.Laplacian(image, cv2.CV_64F)
            noise = laplacian.var()
            return noise / 1000.0  # Normalize
        except:
            return 0.0
    
    def estimate_text_size(self, image: np.ndarray) -> float:
        """Estimate average text size"""
        try:
            # Use morphological operations to estimate text size
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            opened = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
            
            # Find contours
            contours, _ = cv2.findContours(opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                # Calculate average contour area
                areas = [cv2.contourArea(c) for c in contours]
                avg_area = np.mean(areas)
                return np.sqrt(avg_area)
            
            return 20.0  # Default
        except:
            return 20.0
    
    def estimate_skew(self, image: np.ndarray) -> float:
        """Estimate text skew angle"""
        try:
            # Use Hough transform to detect lines
            edges = cv2.Canny(image, 50, 150)
            lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=100)
            
            if lines is not None:
                angles = []
                for line in lines:
                    rho, theta = line[0]
                    angle = theta * 180 / np.pi
                    if 45 < angle < 135:  # Horizontal lines
                        angles.append(angle - 90)
                
                if angles:
                    return np.median(angles)
            
            return 0.0
        except:
            return 0.0
    
    def apply_adaptive_preprocessing(self, image: np.ndarray) -> np.ndarray:
        """Apply adaptive preprocessing based on analysis"""
        analysis = self.analyze_document(image)
        
        processed = image.copy()
        
        # Apply contrast enhancement if needed
        if analysis['contrast_level'] == 'low':
            processed = self.enhance_contrast(processed)
        
        # Apply noise reduction if needed
        if analysis['noise_level'] == 'high':
            processed = self.reduce_noise(processed)
        
        # Apply text size enhancement if needed
        if analysis['text_size'] == 'small':
            processed = self.upscale_text(processed)
        
        # Apply skew correction if needed
        if abs(analysis['skew_angle']) > 5:
            processed = self.correct_skew(processed, analysis['skew_angle'])
        
        return processed
    
    def enhance_contrast(self, image: np.ndarray) -> np.ndarray:
        """Enhance contrast for low-contrast images"""
        try:
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image.copy()
            
            # Apply CLAHE
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
            enhanced = clahe.apply(gray)
            
            return enhanced
        except:
            return image
    
    def reduce_noise(self, image: np.ndarray) -> np.ndarray:
        """Reduce noise in high-noise images"""
        try:
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image.copy()
            
            # Apply bilateral filter
            denoised = cv2.bilateralFilter(gray, 9, 75, 75)
            
            # Apply median filter
            denoised = cv2.medianBlur(denoised, 3)
            
            return denoised
        except:
            return image
    
    def upscale_text(self, image: np.ndarray) -> np.ndarray:
        """Upscale small text for better OCR"""
        try:
            # Upscale by factor of 2
            height, width = image.shape[:2]
            upscaled = cv2.resize(image, (width * 2, height * 2), interpolation=cv2.INTER_CUBIC)
            
            # Apply sharpening
            kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
            sharpened = cv2.filter2D(upscaled, -1, kernel)
            
            return sharpened
        except:
            return image
    
    def correct_skew(self, image: np.ndarray, angle: float) -> np.ndarray:
        """Correct skew angle"""
        try:
            # Rotate image to correct skew
            corrected = imutils.rotate_bound(image, -angle)
            return corrected
        except:
            return image

# Usage example
def test_accuracy_enhancements():
    """Test accuracy enhancement techniques"""
    enhancer = AdvancedAccuracyEnhancements()
    ensemble = MultiModelEnsemble()
    adaptive = AdaptivePreprocessing()
    
    print("🧪 Testing Advanced Accuracy Enhancements...")
    
    # Test with sample image
    sample_image = np.ones((100, 200), dtype=np.uint8) * 255
    cv2.putText(sample_image, "TEST DOCUMENT", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, 0, 2)
    
    # Test enhancement
    result = enhancer.enhance_document_accuracy(sample_image, 'pan_card')
    print(f"Enhancement result: {result.confidence_boost:.2f} boost")
    print(f"Enhancements applied: {result.enhancement_applied}")
    
    # Test ensemble OCR
    ensemble_result = ensemble.ensemble_extract(sample_image)
    print(f"Ensemble OCR: {ensemble_result['best_confidence']:.2f} confidence")
    
    # Test adaptive preprocessing
    analysis = adaptive.analyze_document(sample_image)
    print(f"Document analysis: {analysis}")

if __name__ == "__main__":
    test_accuracy_enhancements()
