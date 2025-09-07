import streamlit as st
import numpy as np
from PIL import Image
import cv2
import os
from deepfake_detector import DeepfakeDetector
from utils import preprocess_image, validate_image
from audio_detector import AudioDeepfakeDetector
from video_detector import VideoDeepfakeDetector

# Initialize the deepfake detectors
@st.cache_resource
def load_detectors():
    """Load all deepfake detection models"""
    image_detector = DeepfakeDetector()
    audio_detector = AudioDeepfakeDetector()
    video_detector = VideoDeepfakeDetector()
    return image_detector, audio_detector, video_detector

def main():
    st.set_page_config(
        page_title="Deepfake Detection System",
        page_icon="🔍",
        layout="wide"
    )
    
    st.title("🔍 Deepfake Detection System")
    st.markdown("""
    Upload images, audio files, or videos to detect whether they're real or deepfakes. 
    Our AI system analyzes multiple types of media using advanced computer vision and signal processing techniques.
    """)
    
    # Load the detectors
    try:
        image_detector, audio_detector, video_detector = load_detectors()
    except Exception as e:
        st.error(f"Failed to load detection models: {str(e)}")
        st.stop()
    
    # Media type selection
    st.header("📂 Select Media Type")
    media_type = st.selectbox(
        "Choose the type of media you want to analyze:",
        ["Image", "Audio", "Video"],
        help="Select the type of file you want to check for deepfake content"
    )
    
    # File upload section based on media type
    if media_type == "Image":
        st.header("📤 Upload Image")
        uploaded_file = st.file_uploader(
            "Choose an image file",
            type=['jpg', 'jpeg', 'png'],
            help="Supported formats: JPG, JPEG, PNG"
        )
    elif media_type == "Audio":
        st.header("🎵 Upload Audio")
        uploaded_file = st.file_uploader(
            "Choose an audio file",
            type=['wav', 'mp3', 'ogg'],
            help="Supported formats: WAV, MP3, OGG"
        )
    else:  # Video
        st.header("🎬 Upload Video")
        uploaded_file = st.file_uploader(
            "Choose a video file",
            type=['mp4', 'avi', 'mov', 'mkv'],
            help="Supported formats: MP4, AVI, MOV, MKV"
        )
    
    if uploaded_file is not None:
        if media_type == "Image":
            process_image_analysis(uploaded_file, image_detector)
        elif media_type == "Audio":
            process_audio_analysis(uploaded_file, audio_detector)
        else:  # Video
            process_video_analysis(uploaded_file, video_detector)

def process_image_analysis(uploaded_file, detector):
    """Process image analysis"""
    try:
        # Display the uploaded image
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("📷 Uploaded Image")
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_container_width=True)
            
            # Image info
            st.info(f"""
            **Image Details:**
            - Format: {image.format}
            - Size: {image.size[0]} x {image.size[1]} pixels
            - Mode: {image.mode}
            """)
        
        with col2:
            st.subheader("🔍 Analysis Results")
            
            # Validate image
            if not validate_image(image):
                st.error("Invalid image format or corrupted file. Please upload a valid image.")
                return
            
            # Process the image
            with st.spinner("Analyzing image for deepfake detection..."):
                try:
                    # Preprocess the image
                    processed_image = preprocess_image(image)
                    
                    # Get prediction
                    prediction, confidence = detector.predict(processed_image)
                    
                    # Display results
                    st.success("✅ Analysis Complete!")
                    
                    # Main result
                    if prediction == "REAL":
                        st.success(f"🟢 **REAL IMAGE**")
                        st.success(f"Confidence: {confidence:.1f}%")
                    else:
                        st.error(f"🔴 **DEEPFAKE DETECTED**")
                        st.error(f"Confidence: {confidence:.1f}%")
                    
                    # Confidence meter
                    st.subheader("📊 Confidence Score")
                    progress_value = confidence / 100
                    
                    if prediction == "REAL":
                        st.success(f"Real: {confidence:.1f}%")
                        st.progress(progress_value)
                    else:
                        st.error(f"Deepfake: {confidence:.1f}%")
                        st.progress(progress_value)
                    
                    # Detailed analysis
                    st.subheader("🔬 Detailed Analysis")
                    analysis_details = detector.get_analysis_details(processed_image)
                    
                    for feature, score in analysis_details.items():
                        if score > 0.7:
                            st.success(f"✅ {feature}: {score:.3f}")
                        elif score > 0.4:
                            st.warning(f"⚠️ {feature}: {score:.3f}")
                        else:
                            st.error(f"❌ {feature}: {score:.3f}")
                    
                    # Interpretation guide
                    with st.expander("🤔 How to interpret these results"):
                        st.markdown("""
                        **Confidence Score**: Indicates how certain the model is about its prediction.
                        - **90-100%**: Very high confidence
                        - **70-89%**: High confidence  
                        - **50-69%**: Moderate confidence
                        - **Below 50%**: Low confidence
                        
                        **Feature Analysis**:
                        - **Facial Consistency**: Checks for inconsistencies in facial features
                        - **Edge Detection**: Analyzes image edges for artifacts
                        - **Texture Analysis**: Examines skin and facial texture patterns
                        - **Compression Artifacts**: Detects unusual compression patterns
                        """)
                        
                except Exception as e:
                    st.error(f"Error during analysis: {str(e)}")
                    st.error("Please try uploading a different image or contact support.")
                    
    except Exception as e:
        st.error(f"Error processing uploaded file: {str(e)}")

def process_audio_analysis(uploaded_file, detector):
    """Process audio analysis"""
    try:
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("🎵 Uploaded Audio")
            st.audio(uploaded_file, format="audio/wav")
            
            # Audio info (basic)
            file_size = len(uploaded_file.read()) / 1024  # KB
            uploaded_file.seek(0)  # Reset file pointer
            st.info(f"""
            **Audio Details:**
            - File size: {file_size:.1f} KB
            - Format: {uploaded_file.type if hasattr(uploaded_file, 'type') else 'Unknown'}
            """)
        
        with col2:
            st.subheader("🔍 Analysis Results")
            
            with st.spinner("Analyzing audio for deepfake detection..."):
                try:
                    # Get prediction
                    prediction, confidence, analysis_details = detector.analyze_audio(uploaded_file)
                    
                    # Display results
                    st.success("✅ Analysis Complete!")
                    
                    # Main result
                    if prediction == "REAL":
                        st.success(f"🟢 **REAL AUDIO**")
                        st.success(f"Confidence: {confidence:.1f}%")
                    else:
                        st.error(f"🔴 **DEEPFAKE DETECTED**")
                        st.error(f"Confidence: {confidence:.1f}%")
                    
                    # Confidence meter
                    st.subheader("📊 Confidence Score")
                    progress_value = confidence / 100
                    st.progress(progress_value)
                    
                    # Detailed analysis
                    st.subheader("🔬 Detailed Analysis")
                    for feature, score in analysis_details.items():
                        if isinstance(score, (int, float)):
                            if score > 0.7:
                                st.success(f"✅ {feature}: {score:.3f}")
                            elif score > 0.4:
                                st.warning(f"⚠️ {feature}: {score:.3f}")
                            else:
                                st.error(f"❌ {feature}: {score:.3f}")
                        else:
                            st.info(f"ℹ️ {feature}: {score}")
                    
                    # Interpretation guide
                    with st.expander("🤔 How to interpret audio results"):
                        st.markdown("""
                        **Audio Analysis Features**:
                        - **Voice Naturalness**: Analyzes spectral characteristics of speech
                        - **Pitch Consistency**: Checks for natural pitch variation
                        - **Temporal Consistency**: Examines timing and rhythm patterns
                        - **Frequency Analysis**: Detects unusual frequency artifacts
                        - **Background Noise**: Analyzes noise characteristics
                        """)
                        
                except Exception as e:
                    st.error(f"Error during audio analysis: {str(e)}")
                    st.error("Please try uploading a different audio file.")
                    
    except Exception as e:
        st.error(f"Error processing uploaded audio file: {str(e)}")

def process_video_analysis(uploaded_file, detector):
    """Process video analysis"""
    try:
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("🎬 Uploaded Video")
            st.video(uploaded_file)
            
            # Video info
            video_info = detector.get_video_info(uploaded_file)
            if video_info:
                st.info(f"""
                **Video Details:**
                - Duration: {video_info['duration']:.1f} seconds
                - Resolution: {video_info['width']} x {video_info['height']}
                - FPS: {video_info['fps']:.1f}
                - Total Frames: {video_info['frame_count']}
                """)
            else:
                st.warning("Could not extract video information")
        
        with col2:
            st.subheader("🔍 Analysis Results")
            
            with st.spinner("Analyzing video for deepfake detection... This may take a moment."):
                try:
                    # Get prediction
                    prediction, confidence, analysis_details = detector.analyze_video(uploaded_file)
                    
                    # Display results
                    st.success("✅ Analysis Complete!")
                    
                    # Main result
                    if prediction == "REAL":
                        st.success(f"🟢 **REAL VIDEO**")
                        st.success(f"Confidence: {confidence:.1f}%")
                    else:
                        st.error(f"🔴 **DEEPFAKE DETECTED**")
                        st.error(f"Confidence: {confidence:.1f}%")
                    
                    # Confidence meter
                    st.subheader("📊 Confidence Score")
                    progress_value = confidence / 100
                    st.progress(progress_value)
                    
                    # Video-specific analysis
                    st.subheader("🔬 Video Analysis")
                    summary = analysis_details.get('summary', {})
                    for feature, score in summary.items():
                        if isinstance(score, (int, float)):
                            if score > 0.7:
                                st.success(f"✅ {feature}: {score:.3f}")
                            elif score > 0.4:
                                st.warning(f"⚠️ {feature}: {score:.3f}")
                            else:
                                st.error(f"❌ {feature}: {score:.3f}")
                    
                    # Frame analysis summary
                    total_frames = analysis_details.get('total_frames', 0)
                    if total_frames > 0:
                        st.info(f"Analyzed {total_frames} frames from the video")
                    
                    # Interpretation guide
                    with st.expander("🤔 How to interpret video results"):
                        st.markdown("""
                        **Video Analysis Features**:
                        - **Frame Consistency**: Individual frame deepfake detection results
                        - **Temporal Consistency**: Smoothness of transitions between frames
                        - **Motion Analysis**: Natural movement patterns
                        - **Face Tracking**: Consistency of facial features across frames
                        """)
                        
                except Exception as e:
                    st.error(f"Error during video analysis: {str(e)}")
                    st.error("Please try uploading a different video file.")
                    
    except Exception as e:
        st.error(f"Error processing uploaded video file: {str(e)}")
    
    # Information section
    st.markdown("---")
    st.header("ℹ️ About This System")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("🎯 Accuracy")
        st.info("Our model achieves high accuracy on various deepfake detection benchmarks through advanced neural network architectures.")
    
    with col2:
        st.subheader("⚡ Speed")
        st.info("Fast processing ensures quick results while maintaining detection quality.")
    
    with col3:
        st.subheader("🔒 Privacy")
        st.info("Images are processed locally and not stored on our servers, ensuring your privacy.")
    
    # Technical details
    with st.expander("🔧 Technical Details"):
        st.markdown("""
        **Model Architecture**: Convolutional Neural Network (CNN) with attention mechanisms
        
        **Training Data**: Based on Meta's Deepfake Detection Challenge dataset
        
        **Detection Methods**:
        - **Images**: Facial landmark analysis, compression artifact analysis, texture and edge pattern recognition
        - **Audio**: Voice naturalness analysis, pitch consistency, temporal patterns, frequency domain analysis
        - **Video**: Frame-by-frame analysis, temporal consistency, motion analysis, face tracking
        
        **Supported Formats**: 
        - **Images**: JPG, JPEG, PNG
        - **Audio**: WAV, MP3, OGG
        - **Video**: MP4, AVI, MOV, MKV
        
        **Processing**: Real-time analysis with computer vision and signal processing techniques
        """)

if __name__ == "__main__":
    main()