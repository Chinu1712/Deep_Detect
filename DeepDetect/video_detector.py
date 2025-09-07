import numpy as np
import cv2
import tempfile
import os
# from moviepy.editor import VideoFileClip  # Commenting out due to dependency issues
from deepfake_detector import DeepfakeDetector
from utils import preprocess_image, validate_image
from PIL import Image

class VideoDeepfakeDetector:
    """
    Video deepfake detection by analyzing individual frames and temporal consistency
    """
    
    def __init__(self):
        self.image_detector = DeepfakeDetector()
        self.frame_sample_rate = 1.0  # Analyze every second
        self.max_frames = 30  # Maximum frames to analyze
        
    def analyze_video(self, video_file):
        """
        Analyze video file for deepfake indicators
        
        Args:
            video_file: Video file path or uploaded file
            
        Returns:
            tuple: (prediction_label, confidence_percentage, analysis_details)
        """
        try:
            # Save uploaded file temporarily
            if hasattr(video_file, 'read'):
                with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
                    tmp_file.write(video_file.read())
                    tmp_file.flush()
                    video_path = tmp_file.name
            else:
                video_path = video_file
            
            # Extract frames for analysis
            frames = self._extract_frames(video_path)
            
            if not frames:
                return "REAL", 50.0, {"error": "No frames could be extracted"}
            
            # Analyze individual frames
            frame_results = self._analyze_frames(frames)
            
            # Analyze temporal consistency
            temporal_analysis = self._analyze_temporal_consistency(frames)
            
            # Combine results
            overall_analysis = self._combine_analysis(frame_results, temporal_analysis)
            
            # Calculate final prediction
            overall_score = overall_analysis['overall_score']
            
            if overall_score > 0.5:
                label = "REAL"
                confidence = overall_score * 100
            else:
                label = "DEEPFAKE"
                confidence = (1 - overall_score) * 100
            
            # Clean up temporary file
            if hasattr(video_file, 'read'):
                os.unlink(video_path)
            
            return label, confidence, overall_analysis
            
        except Exception as e:
            return "REAL", 55.0, {"error": f"Video analysis failed: {str(e)}"}
    
    def _extract_frames(self, video_path):
        """Extract frames from video for analysis"""
        try:
            frames = []
            cap = cv2.VideoCapture(video_path)
            
            if not cap.isOpened():
                return frames
            
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = total_frames / fps if fps > 0 else 0
            
            # Calculate frame sampling
            if duration > 0:
                frame_interval = max(1, int(fps * self.frame_sample_rate))
                max_frames_from_duration = min(self.max_frames, int(duration / self.frame_sample_rate))
            else:
                frame_interval = 1
                max_frames_from_duration = self.max_frames
            
            frame_count = 0
            extracted_count = 0
            
            while cap.isOpened() and extracted_count < max_frames_from_duration:
                ret, frame = cap.read()
                if not ret:
                    break
                
                if frame_count % frame_interval == 0:
                    # Convert BGR to RGB
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frames.append(frame_rgb)
                    extracted_count += 1
                
                frame_count += 1
            
            cap.release()
            return frames
            
        except Exception as e:
            return []
    
    def _analyze_frames(self, frames):
        """Analyze individual frames for deepfake indicators"""
        frame_results = []
        face_consistency_scores = []
        
        for i, frame in enumerate(frames):
            try:
                # Convert to PIL Image for processing
                pil_image = Image.fromarray(frame)
                
                # Validate and preprocess
                if validate_image(pil_image):
                    processed_frame = preprocess_image(pil_image)
                    
                    # Get prediction from image detector
                    label, confidence = self.image_detector.predict(processed_frame)
                    
                    # Get detailed analysis
                    details = self.image_detector.get_analysis_details(processed_frame)
                    
                    frame_result = {
                        'frame_number': i,
                        'prediction': label,
                        'confidence': confidence,
                        'details': details
                    }
                    
                    frame_results.append(frame_result)
                    
                    # Track face consistency
                    face_score = details.get('Facial Consistency', 0.5)
                    face_consistency_scores.append(face_score)
                
            except Exception as e:
                # Skip problematic frames
                continue
        
        return {
            'individual_frames': frame_results,
            'face_consistency_scores': face_consistency_scores,
            'total_frames_analyzed': len(frame_results)
        }
    
    def _analyze_temporal_consistency(self, frames):
        """Analyze temporal consistency between frames"""
        if len(frames) < 2:
            return {'temporal_score': 0.5, 'motion_consistency': 0.5}
        
        try:
            # Calculate optical flow between consecutive frames
            flow_magnitudes = []
            face_movements = []
            
            for i in range(len(frames) - 1):
                frame1_gray = cv2.cvtColor(frames[i], cv2.COLOR_RGB2GRAY)
                frame2_gray = cv2.cvtColor(frames[i + 1], cv2.COLOR_RGB2GRAY)
                
                # Calculate optical flow
                flow = cv2.calcOpticalFlowPyrLK(
                    frame1_gray, frame2_gray,
                    np.array([[100, 100]], dtype=np.float32),  # Simple point
                    None
                )
                
                if flow[0] is not None:
                    magnitude = np.linalg.norm(flow[0] - np.array([[100, 100]]))
                    flow_magnitudes.append(magnitude)
                
                # Analyze face region movement
                face_movement = self._analyze_face_movement(frames[i], frames[i + 1])
                face_movements.append(face_movement)
            
            # Calculate temporal consistency scores
            motion_consistency = self._calculate_motion_consistency(flow_magnitudes)
            face_temporal_score = self._calculate_face_temporal_score(face_movements)
            
            temporal_score = (motion_consistency + face_temporal_score) / 2
            
            return {
                'temporal_score': temporal_score,
                'motion_consistency': motion_consistency,
                'face_temporal_consistency': face_temporal_score,
                'flow_magnitudes': flow_magnitudes
            }
            
        except Exception as e:
            return {'temporal_score': 0.5, 'motion_consistency': 0.5}
    
    def _analyze_face_movement(self, frame1, frame2):
        """Analyze face movement between two frames"""
        try:
            # Convert to grayscale
            gray1 = cv2.cvtColor(frame1, cv2.COLOR_RGB2GRAY)
            gray2 = cv2.cvtColor(frame2, cv2.COLOR_RGB2GRAY)
            
            # Detect faces in both frames
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            faces1 = face_cascade.detectMultiScale(gray1, 1.1, 4)
            faces2 = face_cascade.detectMultiScale(gray2, 1.1, 4)
            
            if len(faces1) == 0 or len(faces2) == 0:
                return 0.5
            
            # Calculate face center movement
            center1 = np.array([faces1[0][0] + faces1[0][2]//2, faces1[0][1] + faces1[0][3]//2])
            center2 = np.array([faces2[0][0] + faces2[0][2]//2, faces2[0][1] + faces2[0][3]//2])
            
            movement = np.linalg.norm(center2 - center1)
            
            # Normalize movement (reasonable movement for natural video)
            if movement < 20:  # Small natural movement
                return 0.8
            elif movement < 50:  # Moderate movement
                return 0.6
            else:  # Large movement (potentially unnatural)
                return 0.3
                
        except Exception as e:
            return 0.5
    
    def _calculate_motion_consistency(self, flow_magnitudes):
        """Calculate motion consistency score"""
        if not flow_magnitudes:
            return 0.5
        
        # Natural videos have relatively consistent motion
        motion_std = np.std(flow_magnitudes)
        motion_mean = np.mean(flow_magnitudes)
        
        # Score based on motion stability
        if motion_std < motion_mean * 0.5:  # Consistent motion
            return 0.8
        elif motion_std < motion_mean:  # Moderately consistent
            return 0.6
        else:  # Inconsistent motion
            return 0.3
    
    def _calculate_face_temporal_score(self, face_movements):
        """Calculate face temporal consistency score"""
        if not face_movements:
            return 0.5
        
        movement_std = np.std(face_movements)
        
        # Natural face movement should be relatively smooth
        if movement_std < 0.2:
            return 0.8
        elif movement_std < 0.4:
            return 0.6
        else:
            return 0.3
    
    def _combine_analysis(self, frame_results, temporal_analysis):
        """Combine frame and temporal analysis results"""
        # Calculate frame-based scores
        frame_predictions = [result['prediction'] for result in frame_results['individual_frames']]
        frame_confidences = [result['confidence'] for result in frame_results['individual_frames']]
        
        if frame_predictions:
            real_count = frame_predictions.count('REAL')
            fake_count = frame_predictions.count('DEEPFAKE')
            frame_ratio = real_count / len(frame_predictions)
            avg_confidence = np.mean(frame_confidences)
        else:
            frame_ratio = 0.5
            avg_confidence = 50.0
        
        # Get temporal score
        temporal_score = temporal_analysis.get('temporal_score', 0.5)
        
        # Combine scores (weighted average)
        overall_score = (frame_ratio * 0.7 + temporal_score * 0.3)
        
        return {
            'overall_score': overall_score,
            'frame_analysis': frame_results,
            'temporal_analysis': temporal_analysis,
            'frame_ratio_real': frame_ratio,
            'average_confidence': avg_confidence,
            'total_frames': len(frame_results['individual_frames']),
            'summary': {
                'Frame Consistency': frame_ratio,
                'Temporal Consistency': temporal_score,
                'Motion Analysis': temporal_analysis.get('motion_consistency', 0.5),
                'Face Tracking': temporal_analysis.get('face_temporal_consistency', 0.5)
            }
        }
    
    def get_video_info(self, video_file):
        """Get basic video information"""
        try:
            if hasattr(video_file, 'read'):
                with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
                    tmp_file.write(video_file.read())
                    tmp_file.flush()
                    video_path = tmp_file.name
            else:
                video_path = video_file
            
            cap = cv2.VideoCapture(video_path)
            
            if not cap.isOpened():
                return None
            
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            duration = frame_count / fps if fps > 0 else 0
            
            cap.release()
            
            if hasattr(video_file, 'read'):
                os.unlink(video_path)
            
            return {
                'fps': fps,
                'frame_count': frame_count,
                'width': width,
                'height': height,
                'duration': duration
            }
            
        except Exception as e:
            return None