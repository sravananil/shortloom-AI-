#!/usr/bin/env python3
"""
ShortLoom Backend - AI-Powered Video Shorts Generator
Detects faces, motion, and energy to create viral-ready short clips
"""

import os
import cv2
import numpy as np
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename
import face_recognition
from moviepy.editor import VideoFileClip, ImageClip, CompositeVideoClip
import tempfile
import shutil
from datetime import datetime
import threading
import time
from vosk import Model, KaldiRecognizer
import wave
import json
import yt_dlp
import uuid
import re
import random
from collections import Counter
from scenedetect import VideoManager, SceneManager
from scenedetect.detectors import ContentDetector
import spacy

# New imports for AI functionality
import requests
from dotenv import load_dotenv

# --- Step 2: Load environment variables right after imports ---
load_dotenv()
HUGGING_FACE_API_KEY = os.getenv("HUGGING_FACE_API_KEY")
API_URL = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.2"

# --- Step 3: Initialize the Flask App ---
app = Flask(__name__)
CORS(app)

# Configuration
UPLOAD_FOLDER = 'uploads'
STATIC_FOLDER = 'static/clips'
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv', 'wmv', 'flv'}
MAX_CONTENT_LENGTH = 500 * 1024 * 1024  # 500MB max file size

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH

# Ensure directories exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(STATIC_FOLDER, exist_ok=True)

# --- Step 4: Define all your functions ---

def generate_title(transcript):
    """Generate a simple catchy title from transcript using keyword frequency."""
    words = re.findall(r'\w+', transcript.lower())
    common = Counter(words).most_common(3)
    title = " ".join([w[0].capitalize() for w in common])
    return f"{title} Moments!"

def generate_hashtags(transcript):
    """Generate hashtags from transcript keywords."""
    words = set(re.findall(r'\w+', transcript.lower()))
    stopwords = {"the", "and", "is", "in", "to", "of", "a", "for", "on", "with", "as", "at", "by", "an", "be", "this", "that"}
    keywords = [w for w in words if w not in stopwords and len(w) > 3]
    hashtags = [f"#{w}" for w in keywords]
    return random.sample(hashtags, min(5, len(hashtags)))

def generate_ai_content(transcript, platform):
    """
    Generates title, description, and hashtags using a Hugging Face model.
    """
    if not HUGGING_FACE_API_KEY:
        print("⚠️ Hugging Face API Key not found. Falling back to basic generation.")
        title = generate_title(transcript)
        hashtags = generate_hashtags(transcript)
        return {"title": title, "description": "Check out this cool clip!", "hashtags": hashtags}

    prompt = create_ai_prompt(transcript, platform)
    headers = {"Authorization": f"Bearer {HUGGING_FACE_API_KEY}"}
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": 250,
            "temperature": 0.7,
            "return_full_text": False,
        }
    }
    
    try:
        print(f"🤖 Calling Hugging Face AI for {platform} content...")
        response = requests.post(API_URL, headers=headers, json=payload, timeout=20)
        response.raise_for_status()
        generated_text = response.json()[0]['generated_text']
        print("✅ AI response received.")
        return parse_ai_response(generated_text)
        
    except requests.exceptions.RequestException as e:
        print(f"❌ Error calling Hugging Face API: {e}")
        title = generate_title(transcript)
        hashtags = generate_hashtags(transcript)
        return {"title": title, "description": "Enjoy this clip!", "hashtags": hashtags}

def create_ai_prompt(transcript, platform):
    """Creates a specific prompt for the AI based on the platform."""
    instruction = "You are a witty social media expert. Based on the following video transcript, generate a title, description, and hashtags. Format your response ONLY as JSON with keys 'title', 'description', and 'hashtags' (which should be an array of strings)."
    if platform == 'tiktok':
        requirements = "The title must be short and punchy (under 100 chars). The description should be brief. Provide 5-7 trending hashtags."
    elif platform == 'instagram':
        requirements = "The title can be a bit longer. The description should be engaging and ask a question. Provide 10-15 relevant hashtags."
    else: # youtube
        requirements = "The title must be SEO-friendly and compelling. The description should be detailed (2-3 sentences), and include a call to action. Provide 3-5 broad hashtags."
    return f"""
    [INST] {instruction}
    Platform: {platform}
    Requirements: {requirements}
    Transcript: "{transcript}"
    [/INST]
    """

def parse_ai_response(text):
    """Parses the JSON string from the AI's response."""
    try:
        json_match = re.search(r'\{.*\}', text, re.DOTALL)
        if json_match:
            json_str = json_match.group(0)
            return json.loads(json_str)
        else:
            print("⚠️ AI response was not valid JSON.")
            return {"title": "AI Generation Failed", "description": text, "hashtags": []}
    except json.JSONDecodeError:
        print("⚠️ Failed to decode AI JSON response.")
        return {"title": "AI Parsing Failed", "description": text, "hashtags": []}

def allowed_file(filename):
    """Check if uploaded file has allowed extension"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

class VideoAnalyzer:
    """Advanced video analysis for detecting engaging moments"""
    
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.motion_threshold = 30
        self.min_clip_duration = 30
        self.max_clip_duration = 45
        self.target_clips = 5
    
    def analyze_video(self, video_path):
        print(f"🎬 Analyzing video: {video_path}")
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened(): raise ValueError("Could not open video file")
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps
        print(f"📊 Video stats: {duration:.1f}s, {fps:.1f} FPS, {total_frames} frames")
        frame_scores, prev_gray, frame_count = [], None, 0
        sample_rate = max(1, int(fps // 5))
        while True:
            ret, frame = cap.read()
            if not ret: break
            frame_count += 1
            if frame_count % sample_rate != 0: continue
            height, width = frame.shape[:2]
            scale = min(640 / width, 480 / height)
            frame_resized = cv2.resize(frame, (int(width * scale), int(height * scale)))
            gray = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)
            score = self.calculate_frame_score(frame_resized, gray, prev_gray)
            frame_scores.append((frame_count / fps, score))
            prev_gray = gray
            if frame_count % (total_frames // 10) == 0: print(f"🔄 Analysis progress: {(frame_count / total_frames) * 100:.1f}%")
        cap.release()
        print(f"✅ Analysis complete! Processed {len(frame_scores)} sample frames")
        return self.find_best_segments(frame_scores, duration, video_path)
    
    def calculate_frame_score(self, frame, gray, prev_gray):
        score = 0
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(30, 30))
        if len(faces) > 0:
            score += 40
            if len(faces) > 1: score += 20
            for (x, y, w, h) in faces: score += min(cv2.Laplacian(gray[y:y+h, x:x+w], cv2.CV_64F).var() / 100, 20)
        if prev_gray is not None:
            flow = cv2.calcOpticalFlowPyrLK(prev_gray, gray, np.array([[x, y] for x in range(0, gray.shape[1], 20) for y in range(0, gray.shape[0], 20)], dtype=np.float32).reshape(-1, 1, 2), None)[0]
            if flow is not None: score += min(np.mean(np.linalg.norm(flow.reshape(-1, 2), axis=1)) * 2, 30)
        score += min(cv2.Laplacian(gray, cv2.CV_64F).var() / 50, 20)
        if 50 < np.mean(gray) < 200: score += 5
        if np.std(gray) > 30: score += 5
        return score
    
    def find_best_segments(self, frame_scores, total_duration, video_path):
        if not frame_scores: return []
        print(f"🎯 Finding best segments from {len(frame_scores)} analyzed frames")
        timestamps, scores = np.array([s[0] for s in frame_scores]), np.array([s[1] for s in frame_scores])
        window_size = min(len(scores) // 10, 5)
        smoothed_scores = np.convolve(scores, np.ones(window_size)/window_size, 'same') if window_size > 1 else scores
        segments, used_ranges = [], []
        from scipy.signal import find_peaks
        try:
            peaks, _ = find_peaks(smoothed_scores, height=np.percentile(smoothed_scores, 70), distance=len(smoothed_scores) // (self.target_clips * 2))
        except ImportError:
            peaks = [i for i in range(1, len(smoothed_scores) - 1) if smoothed_scores[i] > smoothed_scores[i-1] and smoothed_scores[i] > smoothed_scores[i+1] and smoothed_scores[i] > np.percentile(smoothed_scores, 75)]
        print(f"🔍 Found {len(peaks)} potential peak moments")
        for peak_idx in peaks:
            if peak_idx >= len(timestamps): continue
            center_time, peak_score = timestamps[peak_idx], smoothed_scores[peak_idx]
            if peak_score < 70: continue
            clip_duration = 40 if peak_score >= 80 else 30
            start_time = max(0, center_time - clip_duration / 2)
            end_time = min(total_duration, start_time + clip_duration)
            if any(not (end_time <= s or start_time >= e) for s, e in used_ranges): continue
            if end_time - start_time >= 10:
                segments.append((start_time, end_time, peak_score))
                used_ranges.append((start_time, end_time))
        segments.sort(key=lambda x: x[2], reverse=True)
        segments = segments[:self.target_clips]
        segments.sort(key=lambda x: x[0])
        print(f"✨ Selected {len(segments)} best segments:")
        for i, (s, e, sc) in enumerate(segments, 1): print(f"   Clip {i}: {s:.1f}s - {e:.1f}s (score: {sc:.1f})")
        return segments

def create_clip(video_path, start_time, end_time, output_path):
    try:
        print(f"✂️  Creating clip: {start_time:.1f}s - {end_time:.1f}s")
        with VideoFileClip(video_path) as video:
            with video.subclip(start_time, end_time) as clip:
                w, h = clip.size
                final_clip = clip.crop(x_center=w/2, y_center=h/2, width=min(w,h), height=min(w,h)).resize((1080, 1920)) if w > h else clip
                final_clip.write_videofile(output_path, codec='libx264', audio_codec='aac', temp_audiofile='temp-audio.m4a', remove_temp=True, verbose=False, logger=None)
        print(f"✅ Clip saved: {output_path}")
        return True
    except Exception as e:
        print(f"❌ Error creating clip: {e}")
        return False

def add_watermark(input_path, output_path, logo_path="shortloom_watermark.png", pos=("right", "bottom")):
    with VideoFileClip(input_path) as video:
        logo = (ImageClip(logo_path).set_duration(video.duration).resize(height=int(video.h * 0.05)).margin(right=30, bottom=30, opacity=0).set_pos(pos).set_opacity(0.7))
        with CompositeVideoClip([video, logo]) as final:
            final.write_videofile(output_path, codec="libx264", audio_codec="aac")

def extract_audio(video_path, audio_path):
    with VideoFileClip(video_path) as clip:
        if clip.audio: clip.audio.write_audiofile(audio_path)
        else: print("⚠️  No audio track found."); open(audio_path, "wb").close()

def generate_srt(audio_path, srt_path, model_path="vosk-model-small-en-us-0.15"):
    model_path = os.path.join(os.path.dirname(__file__), model_path)
    if not os.path.exists(model_path):
        print(f"Vosk model not found at {model_path}. Please download and extract it.")
        return
    model = Model(model_path)
    with wave.open(audio_path, "rb") as wf:
        rec = KaldiRecognizer(model, wf.getframerate())
        rec.SetWords(True)
        results = []
        while True:
            data = wf.readframes(4000)
            if len(data) == 0: break
            if rec.AcceptWaveform(data): results.append(json.loads(rec.Result()))
        results.append(json.loads(rec.FinalResult()))
    with open(srt_path, "w") as f:
        idx = 1
        for res in results:
            if res.get("result"):
                for word in res["result"]:
                    f.write(f"{idx}\n{format_time(word['start'])} --> {format_time(word['end'])}\n{word['word']}\n\n"); idx += 1

def format_time(seconds):
    h, rem = divmod(seconds, 3600); m, s = divmod(rem, 60)
    return f"{int(h):02}:{int(m):02}:{int(s):02},{int((s - int(s)) * 1000):03}"

def crop_aspect_ratio(clip, aspect_ratio='9:16'):
    w, h = clip.size
    ratios = {'9:16': 9/16, '16:9': 16/9, '1:1': 1}
    target_ratio = ratios.get(aspect_ratio, w/h)
    new_w, new_h = (int(h * target_ratio), h) if w/h > target_ratio else (w, int(w / target_ratio))
    return clip.crop(x_center=w/2, y_center=h/2, width=new_w, height=new_h)

def cleanup_old_files():
    while True:
        time.sleep(3600)
        for folder in [UPLOAD_FOLDER, STATIC_FOLDER]:
            for filename in os.listdir(folder):
                file_path = os.path.join(folder, filename)
                if os.path.isfile(file_path) and os.path.getmtime(file_path) < time.time() - 86400:
                    os.remove(file_path); print(f"🗑️  Cleaned up old file: {filename}")

# --- Step 5: Define Flask Routes ---

@app.route('/')
def home():
    return jsonify({"status": "ShortLoom Backend is running! 🚀", "version": "1.0.0"})

@app.route('/upload', methods=['POST'])
def upload_video():
    if 'video' not in request.files: return jsonify({"error": "No video file provided"}), 400
    file = request.files['video']
    if file.filename == '' or not allowed_file(file.filename): return jsonify({"error": "File not selected or type not allowed"}), 400
    
    filename = secure_filename(file.filename)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    video_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{timestamp}_{filename}")
    file.save(video_path)
    
    try:
        with VideoFileClip(video_path) as clip:
            with crop_aspect_ratio(clip, request.form.get('aspect_ratio', '9:16')) as cropped_clip:
                cropped_video_path = video_path.replace(".mp4", "_cropped.mp4")
                cropped_clip.write_videofile(cropped_video_path, codec='libx264', audio_codec='aac')
        
        segments = VideoAnalyzer().analyze_video(cropped_video_path)
        if not segments: return jsonify({"error": "Could not find any suitable segments"}), 400
        
        clip_urls, full_transcript = [], ""
        for i, (start, end, score) in enumerate(segments, 1):
            clip_path = os.path.join(STATIC_FOLDER, f"{timestamp}_clip_{i}.mp4")
            if create_clip(cropped_video_path, start, end, clip_path):
                watermarked_path = clip_path.replace('.mp4', '_wm.mp4')
                add_watermark(clip_path, watermarked_path)
                os.remove(clip_path)
                audio_path = watermarked_path.replace('.mp4', '.wav')
                srt_path = watermarked_path.replace('.mp4', '.srt')
                extract_audio(watermarked_path, audio_path)
                generate_srt(audio_path, srt_path)
                try:
                    with open(srt_path, "r", encoding="utf-8") as f:
                        full_transcript += " ".join([l.strip() for l in f if not re.match(r'^\d+$|-->', l)]) + " "
                except Exception as e: print(f"⚠️ Could not read SRT: {e}")
                os.remove(audio_path)
                clip_urls.append({"url": f"/static/clips/{os.path.basename(watermarked_path)}", "filename": os.path.basename(watermarked_path), "start_time": round(start, 1), "end_time": round(end, 1), "duration": round(end - start, 1), "score": round(score, 1), "srtUrl": f"/static/clips/{os.path.basename(srt_path)}"})
        
        os.remove(video_path)
        os.remove(cropped_video_path)
        
        platform = request.form.get('platform', 'tiktok')
        ai_meta = generate_ai_content(full_transcript.strip(), platform) if full_transcript.strip() else {"title": "Exciting Video Clip!", "description": "Check out this moment!", "hashtags": ["#short", "#clip"]}
        
        return jsonify({"success": True, "clips": clip_urls, **ai_meta})
    except Exception as e:
        print(f"❌ Error processing video: {e}")
        return jsonify({"error": f"Processing failed: {str(e)}"}), 500

@app.route('/youtube-upload', methods=['POST'])
def youtube_upload():
    url = request.get_json().get('url')
    if not url: return jsonify({'error': 'No URL provided'}), 400
    temp_video_path = os.path.join(UPLOAD_FOLDER, f"yt_{uuid.uuid4()}.mp4")
    try:
        with yt_dlp.YoutubeDL({'format': 'b[ext=mp4][height<=720]/b[ext=mp4]/b', 'outtmpl': temp_video_path, 'quiet': True, 'noplaylist': True}) as ydl: ydl.download([url])
        
        segments = VideoAnalyzer().analyze_video(temp_video_path)
        if not segments: return jsonify({"error": "Could not find suitable segments"}), 400
        
        clip_urls, full_transcript = [], ""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        for i, (start, end, score) in enumerate(segments, 1):
            clip_path = os.path.join(STATIC_FOLDER, f"yt_{timestamp}_clip_{i}.mp4")
            if create_clip(temp_video_path, start, end, clip_path):
                audio_path, srt_path = clip_path.replace('.mp4', '.wav'), clip_path.replace('.mp4', '.srt')
                extract_audio(clip_path, audio_path)
                generate_srt(audio_path, srt_path)
                try:
                    with open(srt_path, "r", encoding="utf-8") as f:
                        full_transcript += " ".join([l.strip() for l in f if not re.match(r'^\d+$|-->', l)]) + " "
                except Exception as e: print(f"⚠️ Could not read SRT: {e}")
                os.remove(audio_path)
                clip_urls.append({"url": f"/static/clips/{os.path.basename(clip_path)}", "filename": os.path.basename(clip_path), "start_time": round(start, 1), "end_time": round(end, 1), "duration": round(end - start, 1), "score": round(score, 1), "srtUrl": f"/static/clips/{os.path.basename(srt_path)}"})
        
        os.remove(temp_video_path)
        ai_meta = generate_ai_content(full_transcript.strip(), 'youtube') if full_transcript.strip() else {"title": "YouTube Highlights!", "description": "Best moments from the video.", "hashtags": ["#youtube", "#shorts"]}
        return jsonify({"success": True, "clips": clip_urls, **ai_meta})
    except Exception as e:
        if os.path.exists(temp_video_path): os.remove(temp_video_path)
        return jsonify({'error': f'Failed to import video: {str(e)}'}), 500

@app.route('/static/clips/<path:filename>')
def serve_clip(filename):
    return send_from_directory(STATIC_FOLDER, filename)

@app.route('/health')
def health_check():
    return jsonify({"status": "healthy", "timestamp": datetime.now().isoformat()})

# --- Step 6: Main execution block ---
if __name__ == '__main__':
    print("🚀 Starting ShortLoom Backend...")
    cleanup_thread = threading.Thread(target=cleanup_old_files, daemon=True)
    cleanup_thread.start()
    app.run(host='0.0.0.0', port=5000, debug=True, threaded=True)


def generate_title(transcript):
    """Generate a simple catchy title from transcript using keyword frequency."""
    words = re.findall(r'\w+', transcript.lower())
    common = Counter(words).most_common(3)
    title = " ".join([w[0].capitalize() for w in common])
    return f"{title} Moments!"

def generate_hashtags(transcript):
    """Generate hashtags from transcript keywords."""
    words = set(re.findall(r'\w+', transcript.lower()))
    stopwords = {"the", "and", "is", "in", "to", "of", "a", "for", "on", "with", "as", "at", "by", "an", "be", "this", "that"}
    keywords = [w for w in words if w not in stopwords and len(w) > 3]
    hashtags = [f"#{w}" for w in keywords]
    return random.sample(hashtags, min(5, len(hashtags)))

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for frontend communication

# Configuration
UPLOAD_FOLDER = 'uploads'
STATIC_FOLDER = 'static/clips'
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv', 'wmv', 'flv'}
MAX_CONTENT_LENGTH = 500 * 1024 * 1024  # 500MB max file size

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH

# Ensure directories exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(STATIC_FOLDER, exist_ok=True)

# Load SpaCy model for NLP tasks
nlp = spacy.load("en_core_web_sm")

def allowed_file(filename):
    """Check if uploaded file has allowed extension"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

class VideoAnalyzer:
    """Advanced video analysis for detecting engaging moments"""
    
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.motion_threshold = 30
        self.min_clip_duration = 30  # seconds
        self.max_clip_duration = 45  # seconds
        self.target_clips = 5
    
    def analyze_video(self, video_path):
        """
        Analyze video to find the best moments for short clips
        Returns list of (start_time, end_time, score) tuples
        """
        print(f"🎬 Analyzing video: {video_path}")
        
        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError("Could not open video file")
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps
        
        print(f"📊 Video stats: {duration:.1f}s, {fps:.1f} FPS, {total_frames} frames")
        
        # Analysis data
        frame_scores = []
        prev_gray = None
        frame_count = 0
        
        # Sample every 5th frame for performance
        sample_rate = max(1, int(fps // 5))
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Skip frames for performance
            if frame_count % sample_rate != 0:
                continue
            
            # Resize for faster processing
            height, width = frame.shape[:2]
            scale = min(640 / width, 480 / height)
            new_width = int(width * scale)
            new_height = int(height * scale)
            frame_resized = cv2.resize(frame, (new_width, new_height))
            
            # Convert to grayscale
            gray = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)
            
            # Calculate frame score based on multiple factors
            score = self.calculate_frame_score(frame_resized, gray, prev_gray)
            
            timestamp = frame_count / fps
            frame_scores.append((timestamp, score))
            
            prev_gray = gray
            
            # Progress indicator
            if frame_count % (total_frames // 10) == 0:
                progress = (frame_count / total_frames) * 100
                print(f"🔄 Analysis progress: {progress:.1f}%")
        
        cap.release()
        
        print(f"✅ Analysis complete! Processed {len(frame_scores)} sample frames")
        
        # Find best segments
        segments = self.find_best_segments(frame_scores, duration, video_path)
        
        return segments
    
    def calculate_frame_score(self, frame, gray, prev_gray):
        """
        Calculate engagement score for a frame based on:
        - Face detection and activity
        - Motion detection
        - Visual complexity
        """
        score = 0
        
        # 1. Face Detection (40% weight)
        faces = self.face_cascade.detectMultiScale(
            gray, 
            scaleFactor=1.1, 
            minNeighbors=5, 
            minSize=(30, 30)
        )
        
        if len(faces) > 0:
            score += 40
            # Bonus for multiple faces (social interaction)
            if len(faces) > 1:
                score += 20
            
            # Analyze face regions for activity
            for (x, y, w, h) in faces:
                face_roi = gray[y:y+h, x:x+w]
                # Face activity based on local contrast
                face_activity = cv2.Laplacian(face_roi, cv2.CV_64F).var()
                score += min(face_activity / 100, 20)
        
        # 2. Motion Detection (30% weight)
        if prev_gray is not None:
            # Calculate optical flow magnitude
            flow = cv2.calcOpticalFlowPyrLK(
                prev_gray, gray, 
                np.array([[x, y] for x in range(0, gray.shape[1], 20) 
                         for y in range(0, gray.shape[0], 20)], dtype=np.float32).reshape(-1, 1, 2),
                None
            )[0]
            
            if flow is not None:
                motion_magnitude = np.mean(np.linalg.norm(flow.reshape(-1, 2), axis=1))
                score += min(motion_magnitude * 2, 30)
        
        # 3. Visual Complexity (20% weight)
        # More complex scenes are often more engaging
        complexity = cv2.Laplacian(gray, cv2.CV_64F).var()
        score += min(complexity / 50, 20)
        
        # 4. Brightness and Contrast (10% weight)
        # Well-lit scenes are more engaging
        brightness = np.mean(gray)
        contrast = np.std(gray)
        
        # Optimal brightness range
        if 50 < brightness < 200:
            score += 5
        
        # Good contrast
        if contrast > 30:
            score += 5
        
        return score
    
    def find_best_segments(self, frame_scores, total_duration, video_path):
        """
        Find the best segments for short clips using sliding window approach
        """
        if not frame_scores:
            return []
        
        print(f"🎯 Finding best segments from {len(frame_scores)} analyzed frames")
        
        # Convert to numpy arrays for easier processing
        timestamps = np.array([score[0] for score in frame_scores])
        scores = np.array([score[1] for score in frame_scores])
        
        # Smooth scores to reduce noise
        window_size = min(len(scores) // 10, 5)
        if window_size > 1:
            smoothed_scores = np.convolve(scores, np.ones(window_size)/window_size, mode='same')
        else:
            smoothed_scores = scores
        
        segments = []
        used_ranges = []

        # Find peaks in the smoothed scores
        from scipy.signal import find_peaks
        try:
            peaks, properties = find_peaks(
                smoothed_scores, 
                height=np.percentile(smoothed_scores, 70),  # Only top 30% scores
                distance=len(smoothed_scores) // (self.target_clips * 2)  # Ensure spacing
            )
        except ImportError:
            # Fallback if scipy is not available
            peaks = self.find_peaks_simple(smoothed_scores)
        
        print(f"🔍 Found {len(peaks)} potential peak moments")
        
        # Create segments around peaks
        for peak_idx in peaks:
            if peak_idx < len(timestamps):
                center_time = timestamps[peak_idx]
                peak_score = smoothed_scores[peak_idx]

                if peak_score < 70:
                    continue

                if peak_score >= 80:
                    clip_duration = 40
                else:
                    clip_duration = 30

                start_time = max(0, center_time - clip_duration / 2)
                end_time = min(total_duration, start_time + clip_duration)

                # Check for overlap with already selected segments
                overlap = False
                for (used_start, used_end) in used_ranges:
                    # If overlap more than 1 second, skip
                    if not (end_time <= used_start or start_time >= used_end):
                        overlap = True
                        break
                if overlap:
                    continue

                if end_time - start_time >= 10:
                    segments.append((start_time, end_time, peak_score))
                    used_ranges.append((start_time, end_time))
        
        # Sort by score and take top segments
        segments.sort(key=lambda x: x[2], reverse=True)
        segments = segments[:self.target_clips]
        
        # Sort by time for final output
        segments.sort(key=lambda x: x[0])
        
        print(f"✨ Selected {len(segments)} best segments:")
        for i, (start, end, score) in enumerate(segments, 1):
            print(f"   Clip {i}: {start:.1f}s - {end:.1f}s (score: {score:.1f})")
        
        return segments
    
    def find_peaks_simple(self, scores):
        """Simple peak finding fallback"""
        peaks = []
        threshold = np.percentile(scores, 75)
        
        for i in range(1, len(scores) - 1):
            if (scores[i] > scores[i-1] and 
                scores[i] > scores[i+1] and 
                scores[i] > threshold):
                peaks.append(i)
        
        return peaks

def create_clip(video_path, start_time, end_time, output_path):
    """
    Create a video clip from start_time to end_time
    """
    try:
        print(f"✂️  Creating clip: {start_time:.1f}s - {end_time:.1f}s")
        
        # Load video
        video = VideoFileClip(video_path)
        
        # Extract subclip
        clip = video.subclip(start_time, end_time)
        
        # Resize to vertical format (9:16 aspect ratio) if needed
        # This makes it perfect for social media
        w, h = clip.size
        if w > h:  # Landscape video
            # Crop to center square first, then resize to 9:16
            crop_size = min(w, h)
            clip = clip.crop(x_center=w/2, y_center=h/2, width=crop_size, height=crop_size)
            # Resize to 9:16 (e.g., 1080x1920)
            clip = clip.resize((1080, 1920))
        
        # Write the clip
        clip.write_videofile(
            output_path,
            codec='libx264',
            audio_codec='aac',
            temp_audiofile='temp-audio.m4a',
            remove_temp=True,
            verbose=False,
            logger=None
        )
        
        # Clean up
        clip.close()
        video.close()
        
        print(f"✅ Clip saved: {output_path}")
        return True
        
    except Exception as e:
        print(f"❌ Error creating clip: {e}")
        return False

def add_watermark(input_path, output_path, logo_path="shortloom_watermark.png", pos=("right", "bottom")):
    video = VideoFileClip(input_path)
    logo = (
        ImageClip(logo_path)
        .set_duration(video.duration)
        .resize(height=int(video.h * 0.05))  # 5% of video height
        .margin(right=30, bottom=30, opacity=0)
        .set_pos(pos)
        .set_opacity(0.7)  # 70% opacity
    )
    final = CompositeVideoClip([video, logo])
    final.write_videofile(output_path, codec="libx264", audio_codec="aac")
    video.close()
    logo.close()
    final.close()

def extract_audio(video_path, audio_path):
    from moviepy.editor import VideoFileClip
    clip = VideoFileClip(video_path)
    if clip.audio is not None:
        clip.audio.write_audiofile(audio_path)
    else:
        print("⚠️  No audio track found in video.")
        # Optionally, create a silent audio file or skip SRT generation
        with open(audio_path, "wb") as f:
            pass  # create an empty file or handle as needed
    clip.close()

def generate_srt(audio_path, srt_path, model_path="vosk-model-small-en-us-0.15"):
    model_path = os.path.join(os.path.dirname(__file__), "vosk-model-small-en-us-0.15")
    model = Model(model_path)
    wf = wave.open(audio_path, "rb")
    rec = KaldiRecognizer(model, wf.getframerate())
    rec.SetWords(True)
    results = []
    while True:
        data = wf.readframes(4000)
        if len(data) == 0:
            break
        if rec.AcceptWaveform(data):
            results.append(json.loads(rec.Result()))
    results.append(json.loads(rec.FinalResult()))

    # Convert results to SRT
    srt = []
    idx = 1
    for res in results:
        if not res.get("result"):
            continue
        for word in res["result"]:
            start = word["start"]
            end = word["end"]
            text = word["word"]
            srt.append(f"{idx}\n{format_time(start)} --> {format_time(end)}\n{text}\n")
            idx += 1
    with open(srt_path, "w") as f:
        f.writelines(srt)

def format_time(seconds):
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds - int(seconds)) * 1000)
    return f"{h:02}:{m:02}:{s:02},{ms:03}"

LIMITS_FILE = "user_limits.json"

def load_limits():
    try:
        with open(LIMITS_FILE) as f:
            return json.load(f)
    except:
        return {}

def save_limits(limits):
    with open(LIMITS_FILE, "w") as f:
        json.dump(limits, f)

def check_limit(ip, max_per_day=3):
    limits = load_limits()
    today = time.strftime("%Y-%m-%d")
    user = limits.get(ip, {"date": today, "count": 0})
    if user["date"] != today:
        user = {"date": today, "count": 0}
    if user["count"] >= max_per_day:
        return False
    user["count"] += 1
    limits[ip] = user
    save_limits(limits)
    return True

@app.route('/')
def home():
    """Health check endpoint"""
    return jsonify({
        "status": "ShortLoom Backend is running! 🚀",
        "version": "1.0.0",
        "endpoints": ["/upload", "/static/clips/<filename>"]
    })

@app.route('/upload', methods=['POST'])
def upload_video():
    """
    Main endpoint for video upload and processing
    Accepts: multipart/form-data with 'video' file
    Returns: JSON with list of generated clip URLs
    """
    # Check if file was uploaded
    if 'video' not in request.files:
        return jsonify({"error": "No video file provided"}), 400
    
    file = request.files['video']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400
    
    if not allowed_file(file.filename):
        return jsonify({"error": "File type not allowed. Please upload MP4, AVI, MOV, MKV, WMV, or FLV"}), 400
    
    # Save uploaded file
    filename = secure_filename(file.filename)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    video_filename = f"{timestamp}_{filename}"
    video_path = os.path.join(app.config['UPLOAD_FOLDER'], video_filename)
    
    print(f"📥 Uploading: {video_filename}")
    file.save(video_path)
    
    # Aspect ratio handling
    aspect_ratio = request.form.get('aspect_ratio', '9:16')
    if aspect_ratio not in ["9:16", "1:1", "16:9"]:
        return jsonify({"error": "Invalid aspect ratio. Allowed values are 9:16, 1:1, 16:9"}), 400

    try:
        # Crop video to desired aspect ratio
        cropped_video_path = video_path.replace(".mp4", "_cropped.mp4")
        clip = VideoFileClip(video_path)
        cropped_clip = crop_aspect_ratio(clip, aspect_ratio)
        cropped_clip.write_videofile(cropped_video_path, codec='libx264', audio_codec='aac')
        clip.close()
        cropped_clip.close()

        # Initialize analyzer
        analyzer = VideoAnalyzer()

        # Analyze video to find best segments
        segments = analyzer.analyze_video(cropped_video_path)

        if not segments:
            return jsonify({"error": "Could not find any suitable segments in the video"}), 400

        # Generate clips
        clip_urls = []

        for i, (start_time, end_time, score) in enumerate(segments, 1):
            if score < 70:
                continue  # Skip low-score clips
            clip_filename = f"{timestamp}_clip_{i}.mp4"
            clip_path = os.path.join(STATIC_FOLDER, clip_filename)

            # Create the clip
            success = create_clip(cropped_video_path, start_time, end_time, clip_path)

            if success:
                # Add watermark to the clip
                watermarked_path = clip_path.replace('.mp4', '_wm.mp4')
                add_watermark(clip_path, watermarked_path, "shortloom_watermark.png")
                os.remove(clip_path)  # Remove unwatermarked version

                # Extract audio and generate SRT
                audio_path = watermarked_path.replace('.mp4', '.wav')
                srt_path = watermarked_path.replace('.mp4', '.srt')
                extract_audio(watermarked_path, audio_path)
                generate_srt(audio_path, srt_path)
                os.remove(audio_path)  # Clean up

                clip_url = f"/static/clips/{os.path.basename(watermarked_path)}"
                clip_urls.append({
                    "url": clip_url,
                    "filename": os.path.basename(watermarked_path),
                    "start_time": round(start_time, 1),
                    "end_time": round(end_time, 1),
                    "duration": round(end_time - start_time, 1),
                    "score": round(score, 1),
                    "srtUrl": f"/static/clips/{os.path.basename(srt_path)}"
                })

        # Clean up original video
        try:
            os.remove(video_path)
            os.remove(cropped_video_path)
        except:
            pass

        print(f"🎉 Successfully generated {len(clip_urls)} clips!")

        # After generating SRT for the first clip, extract transcript and generate metadata
        if clip_urls:
            # Read transcript from SRT file
            srt_path = os.path.join(STATIC_FOLDER, clip_urls[0]["filename"].replace('.mp4', '.srt'))
            transcript = ""
            try:
                with open(srt_path, "r", encoding="utf-8") as f:
                    lines = f.readlines()
                    transcript = " ".join([line.strip() for line in lines if "-->" not in line and not line.strip().isdigit()])
            except Exception as e:
                print(f"⚠️ Could not read SRT for transcript: {e}")

            # Generate metadata
            title = generate_title(transcript)
            hashtags = generate_hashtags(transcript)
        else:
            title = ""
            hashtags = []

        return jsonify({
            "success": True,
            "message": f"Generated {len(clip_urls)} viral-ready shorts!",
            "clips": clip_urls,
            "total_clips": len(clip_urls),
            "title": title,
            "hashtags": hashtags
        })

    except Exception as e:
        print(f"❌ Error processing video: {e}")
        return jsonify({"error": f"Processing failed: {str(e)}"}), 500

def crop_aspect_ratio(clip, aspect_ratio='9:16'):
    """
    Crop a MoviePy VideoFileClip to the given aspect ratio.
    aspect_ratio: '9:16', '16:9', or '1:1'
    """
    w, h = clip.size
    if aspect_ratio == '9:16':
        target_ratio = 9 / 16
    elif aspect_ratio == '16:9':
        target_ratio = 16 / 9
    elif aspect_ratio == '1:1':
        target_ratio = 1
    else:
        target_ratio = w / h  # fallback to original

    # Decide new width and height
    if w / h > target_ratio:
        # Too wide, crop width
        new_w = int(h * target_ratio)
        new_h = h
    else:
        # Too tall, crop height
        new_w = w
        new_h = int(w / target_ratio)

    x1 = (w - new_w) // 2
    y1 = (h - new_h) // 2
    return clip.crop(x1=x1, y1=y1, x2=x1 + new_w, y2=y1 + new_h)

@app.route('/static/clips/<path:filename>')
def serve_clip(filename):
    """Serve generated video clips"""
    return send_from_directory(STATIC_FOLDER, filename)

@app.route('/health')
def health_check():
    """Detailed health check"""
    return jsonify({
        "status": "healthy",
        "opencv_version": cv2.__version__,
        "upload_folder": os.path.exists(UPLOAD_FOLDER),
        "static_folder": os.path.exists(STATIC_FOLDER),
        "timestamp": datetime.now().isoformat()
    })

# Cleanup old files periodically
def cleanup_old_files():
    """Clean up files older than 24 hours"""
    while True:
        try:
            current_time = time.time()
            
            # Clean upload folder
            for filename in os.listdir(UPLOAD_FOLDER):
                file_path = os.path.join(UPLOAD_FOLDER, filename)
                if os.path.isfile(file_path):
                    file_age = current_time - os.path.getctime(file_path)
                    if file_age > 86400:  # 24 hours
                        os.remove(file_path)
                        print(f"🗑️  Cleaned up old upload: {filename}")
            
            # Clean clips folder
            for filename in os.listdir(STATIC_FOLDER):
                file_path = os.path.join(STATIC_FOLDER, filename)
                if os.path.isfile(file_path):
                    file_age = current_time - os.path.getctime(file_path)
                    if file_age > 86400:  # 24 hours
                        os.remove(file_path)
                        print(f"🗑️  Cleaned up old clip: {filename}")
                        
        except Exception as e:
            print(f"⚠️  Cleanup error: {e}")
        
        # Run cleanup every hour
        time.sleep(3600)

@app.route('/youtube-upload', methods=['POST'])
def youtube_upload():
    data = request.get_json()
    url = data.get('url')
    if not url:
        return jsonify({'error': 'No URL provided'}), 400

    # Generate a unique filename
    temp_id = str(uuid.uuid4())
    temp_video_path = os.path.join(UPLOAD_FOLDER, f"yt_{temp_id}.mp4")
    try:
        # Download video using yt-dlp
        ydl_opts = {
            'format': 'best[ext=mp4][height<=720]/best[ext=mp4]/best',
            'outtmpl': temp_video_path,
            'quiet': True,
            'noplaylist': True,
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])

        # Now process the video as usual
        analyzer = VideoAnalyzer()
        segments = analyzer.analyze_video(temp_video_path)
        
        if not segments:
            return jsonify({"error": "Could not find any suitable segments in the video"}), 400

        # Generate clips similar to upload endpoint
        clip_urls = []
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        for i, (start_time, end_time, score) in enumerate(segments, 1):
            if score < 70:
                continue  # Skip low-score clips
            clip_filename = f"yt_{timestamp}_clip_{i}.mp4"
            clip_path = os.path.join(STATIC_FOLDER, clip_filename)

            # Create the clip
            success = create_clip(temp_video_path, start_time, end_time, clip_path)

            if success:
                clip_url = f"/static/clips/{os.path.basename(clip_path)}"
                clip_urls.append({
                    "url": clip_url,
                    "filename": os.path.basename(clip_path),
                    "start_time": round(start_time, 1),
                    "end_time": round(end_time, 1),
                    "duration": round(end_time - start_time, 1),
                    "score": round(score, 1)
                })

        # Clean up temporary video
        if os.path.exists(temp_video_path):
            os.remove(temp_video_path)
            
        return jsonify({
            "success": True,
            "message": f"Generated {len(clip_urls)} clips from YouTube video!",
            "clips": clip_urls,
            "total_clips": len(clip_urls),
            "title": "YouTube Video Highlights",
            "hashtags": ["#shorts", "#viral", "#youtube"]
        }), 200
        
    except Exception as e:
        if os.path.exists(temp_video_path):
            os.remove(temp_video_path)
        return jsonify({'error': f'Failed to import video: {str(e)}'}), 500
    

if __name__ == '__main__':
    print("🚀 Starting ShortLoom Backend...")
    print("🎬 AI-Powered Video Shorts Generator")
    print("📡 Endpoints:")
    print("   POST /upload - Upload video and generate shorts")
    print("   GET /static/clips/<filename> - Download generated clips")
    print("   GET /health - Health check")
    
    # Start cleanup thread
    cleanup_thread = threading.Thread(target=cleanup_old_files, daemon=True)
    cleanup_thread.start()
    
    # Run Flask app
    app.run(
        host='0.0.0.0',  # Accept connections from any IP
        port=5000,       # Default Flask port
        debug=True,      # Enable debug mode for development
        threaded=True    # Handle multiple requests
    )

def detect_scenes(video_path, threshold=27.0):
    """
    Returns a list of (scene_start_sec, scene_end_sec) tuples.
    """
    video_manager = VideoManager([video_path])
    scene_manager = SceneManager()
    scene_manager.add_detector(ContentDetector(threshold=threshold))
    video_manager.start()
    scene_manager.detect_scenes(frame_source=video_manager)
    scene_list = scene_manager.get_scene_list()
    video_manager.release()
    # Convert to seconds
    scenes = [(start.get_seconds(), end.get_seconds()) for start, end in scene_list]
    return scenes


