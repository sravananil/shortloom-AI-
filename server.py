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
import sqlite3
from flask_cors import CORS


from werkzeug.security import generate_password_hash, check_password_hash
# JWT imports
import jwt
from functools import wraps
from datetime import datetime, timedelta

# New imports for AI functionality
import requests
from dotenv import load_dotenv
import whisper
from transformers import pipeline
from googletrans import Translator

# --- Step 2: Load environment variables right after imports ---
load_dotenv()
HUGGING_FACE_API_KEY = os.getenv("HUGGING_FACE_API_KEY")
API_URL = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.2"

# Load models once at startup
whisper_model = whisper.load_model("small")
title_gen = pipeline("text-generation", model="EleutherAI/gpt-neo-1.3B")
hashtag_gen = pipeline("text-generation", model="EleutherAI/gpt-neo-1.3B")
desc_gen = pipeline("text-generation", model="EleutherAI/gpt-neo-1.3B")
translator = Translator()

# --- Step 3: Initialize the Flask App ---

app = Flask(__name__)
CORS(app, supports_credentials=True, expose_headers=["Authorization"])  # Enable CORS for frontend communication

# Secret key for JWT
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'your_secret_key_here')
def token_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        token = None
        if 'Authorization' in request.headers:
            auth_header = request.headers['Authorization']
            if auth_header.startswith('Bearer '):
                token = auth_header.split(' ')[1]
        if not token:
            return jsonify({'error': 'Token is missing!'}), 401
        try:
            data = jwt.decode(token, app.config['SECRET_KEY'], algorithms=["HS256"])
            current_user = data['email']
        except Exception as e:
            return jsonify({'error': 'Token is invalid!'}), 401
        return f(current_user, *args, **kwargs)
    return decorated

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


# --- User Management: SQLite Setup ---
DB_PATH = "shortloom_users.db"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()

# Always initialize DB at startup
init_db()

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
@token_required
def upload_video(current_user):
    if 'video' not in request.files:
        return jsonify({"error": "No video file provided"}), 400
    file = request.files['video']
    if file.filename == '' or not allowed_file(file.filename):
        return jsonify({"error": "File not selected or type not allowed"}), 400

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
        if not segments:
            if os.path.exists(video_path): os.remove(video_path)
            if os.path.exists(cropped_video_path): os.remove(cropped_video_path)
            return jsonify({"error": "Could not find any suitable segments in the video. Try a video with more action or faces."}), 400

        clip_urls, full_transcript = [], ""
        for i, (start, end, score) in enumerate(segments, 1):
            try:
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
                    except Exception as e:
                        print(f"⚠️ Could not read SRT: {e}")
                    os.remove(audio_path)
                    clip_urls.append({"url": f"/static/clips/{os.path.basename(watermarked_path)}", "filename": os.path.basename(watermarked_path), "start_time": round(start, 1), "end_time": round(end, 1), "duration": round(end - start, 1), "score": round(score, 1), "srtUrl": f"/static/clips/{os.path.basename(srt_path)}"})
            except Exception as e:
                print(f"⚠️ Error processing clip {i}: {e}")
                # Continue with other clips if one fails
                continue

        # Clean up input files regardless of success
        if os.path.exists(video_path): os.remove(video_path)
        if os.path.exists(cropped_video_path): os.remove(cropped_video_path)

        if not clip_urls:
            return jsonify({"error": "Failed to generate any clips from the video. Please try again with a different video."}), 400

        platform = request.form.get('platform', 'tiktok')
        ai_meta = generate_ai_content(full_transcript.strip(), platform) if full_transcript.strip() else {"title": "Exciting Video Clip!", "description": "Check out this moment!", "hashtags": ["#short", "#clip"]}

        return jsonify({"success": True, "clips": clip_urls, **ai_meta})
    except Exception as e:
        # Clean up in case of error
        if os.path.exists(video_path): os.remove(video_path)
        if os.path.exists(cropped_video_path): os.remove(cropped_video_path)
        print(f"❌ Error processing video: {e}")
        return jsonify({"error": f"Processing failed: {str(e)}"}), 500

@app.route('/youtube-upload', methods=['POST'])
def youtube_upload():
    url = request.get_json().get('url')
    if not url: return jsonify({'error': 'No URL provided'}), 400
    
    # Validate YouTube URL
    if not re.match(r'^(https?://)?(www\.)?(youtube\.com|youtu\.be)/.+$', url):
        return jsonify({'error': 'Invalid YouTube URL'}), 400
        
    temp_video_path = os.path.join(UPLOAD_FOLDER, f"yt_{uuid.uuid4()}.mp4")
    try:
        # Download with better error handling
        try:
            with yt_dlp.YoutubeDL({'format': 'b[ext=mp4][height<=720]/b[ext=mp4]/b', 'outtmpl': temp_video_path, 'quiet': True, 'noplaylist': True}) as ydl:
                info = ydl.extract_info(url, download=True)
                if not info:
                    return jsonify({'error': 'Failed to extract video information'}), 400
        except yt_dlp.utils.DownloadError as e:
            return jsonify({'error': f'YouTube download failed: {str(e)}'}), 400
            
        if not os.path.exists(temp_video_path):
            return jsonify({'error': 'Download completed but video file not found'}), 500
        
        segments = VideoAnalyzer().analyze_video(temp_video_path)
        if not segments: return jsonify({"error": "Could not find suitable segments in the video. Try a video with more action or faces."}), 400
        
        clip_urls, full_transcript = [], ""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        for i, (start, end, score) in enumerate(segments, 1):
            try:
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
            except Exception as e:
                print(f"⚠️ Error processing clip {i}: {e}")
                # Continue with other clips if one fails
                continue
        
        os.remove(temp_video_path)
        
        if not clip_urls:
            return jsonify({"error": "Failed to generate any clips from the video. Please try again with a different video."}), 400
            
        ai_meta = generate_ai_content(full_transcript.strip(), 'youtube') if full_transcript.strip() else {"title": "YouTube Highlights!", "description": "Best moments from the video.", "hashtags": ["#youtube", "#shorts"]}
        return jsonify({"success": True, "clips": clip_urls, **ai_meta})
    except Exception as e:
        if os.path.exists(temp_video_path): os.remove(temp_video_path)
        print(f"❌ Error processing YouTube video: {str(e)}")
        return jsonify({'error': f'Failed to import video: {str(e)}'}), 500

@app.route('/static/clips/<path:filename>')
def serve_clip(filename):
    return send_from_directory(STATIC_FOLDER, filename)

@app.route('/health')
def health_check():
    return jsonify({"status": "healthy", "timestamp": datetime.now().isoformat()})


# --- Registration Endpoint ---
@app.route('/register', methods=['POST'])
def register():
    data = request.get_json()
    email = data.get('email')
    password = data.get('password')
    if not email or not password:
        return jsonify({'error': 'Email and password required'}), 400
    password_hash = generate_password_hash(password)
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("INSERT INTO users (email, password_hash) VALUES (?, ?)", (email, password_hash))
        conn.commit()
        conn.close()
        print(f"✅ Registered user: {email}")
        return jsonify({'success': True, 'message': 'User registered successfully'})
    except sqlite3.IntegrityError:
        print(f"⚠️ Registration failed: Email already registered: {email}")
        return jsonify({'error': 'Email already registered'}), 409
    except Exception as e:
        print(f"❌ Registration error: {e}")
        return jsonify({'error': 'Registration failed'}), 500


# --- Login Endpoint ---
# JWT Token Configuration
ACCESS_TOKEN_EXPIRY = 60*60*24  # 24 hours in seconds
REFRESH_TOKEN_EXPIRY = 60*60*24*7  # 7 days in seconds

def generate_tokens(email):
    """Generate both access and refresh tokens"""
    access_token = jwt.encode({
        'email': email, 
        'exp': datetime.utcnow().timestamp() + ACCESS_TOKEN_EXPIRY,
        'token_type': 'access'
    }, app.config['SECRET_KEY'], algorithm="HS256")
    
    refresh_token = jwt.encode({
        'email': email, 
        'exp': datetime.utcnow().timestamp() + REFRESH_TOKEN_EXPIRY,
        'token_type': 'refresh'
    }, app.config['SECRET_KEY'], algorithm="HS256")
    
    return access_token, refresh_token

@app.route('/login', methods=['POST'])
def login():
    data = request.get_json()
    email = data.get('email')
    password = data.get('password')
    if not email or not password:
        return jsonify({'error': 'Email and password required'}), 400
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("SELECT password_hash FROM users WHERE email = ?", (email,))
        row = c.fetchone()
        conn.close()
        if row and check_password_hash(row[0], password):
            # Generate both tokens
            access_token, refresh_token = generate_tokens(email)
            print(f"✅ Login success: {email}")
            return jsonify({
                'success': True, 
                'message': 'Login successful', 
                'access_token': access_token,
                'refresh_token': refresh_token
            })
        else:
            print(f"❌ Login failed: Invalid credentials for {email}")
            return jsonify({'error': 'Invalid credentials'}), 401
    except Exception as e:
        print(f"❌ Login error: {e}")
        return jsonify({'error': 'Login failed'}), 500
        
@app.route('/refresh-token', methods=['POST'])
def refresh_token():
    data = request.get_json()
    token = data.get('refresh_token')
    if not token:
        return jsonify({'error': 'Refresh token is required'}), 400
    try:
        # Verify the refresh token
        payload = jwt.decode(token, app.config['SECRET_KEY'], algorithms=["HS256"])
        
        # Check if it's actually a refresh token
        if payload.get('token_type') != 'refresh':
            return jsonify({'error': 'Invalid token type'}), 401
            
        # Generate new tokens
        access_token, new_refresh_token = generate_tokens(payload['email'])
        
        return jsonify({
            'success': True,
            'access_token': access_token,
            'refresh_token': new_refresh_token
        })
    except jwt.ExpiredSignatureError:
        return jsonify({'error': 'Refresh token has expired'}), 401
    except jwt.InvalidTokenError:
        return jsonify({'error': 'Invalid refresh token'}), 401
    except Exception as e:
        print(f"❌ Token refresh error: {e}")
        return jsonify({'error': 'Token refresh failed'}), 500

def process_short(video_path, target_lang="en"):
    # 1. Transcription
    result = whisper_model.transcribe(video_path)
    transcript = result["text"]

    # 2. Translation
    translated = translator.translate(transcript, dest=target_lang).text

    # 3. Title Generation
    title_prompt = f"Generate a catchy YouTube short title for: {translated}"
    title = title_gen(title_prompt, max_length=20)[0]["generated_text"]

    # 4. Hashtag Generation
    hashtag_prompt = f"Suggest 5 relevant hashtags for: {translated}"
    hashtags = hashtag_gen(hashtag_prompt, max_length=30)[0]["generated_text"]

    # 5. Description Generation
    desc_prompt = f"Write a short engaging description for: {translated}"
    description = desc_gen(desc_prompt, max_length=60)[0]["generated_text"]

    return {
        "transcript": transcript,
        "translated": translated,
        "title": title.strip(),
        "hashtags": hashtags.strip(),
        "description": description.strip()
    }

# --- Step 6: Main execution block ---
if __name__ == '__main__':
    print("🚀 Starting ShortLoom Backend...")
    cleanup_thread = threading.Thread(target=cleanup_old_files, daemon=True)
    cleanup_thread.start()
    app.run(host='0.0.0.0', port=5000, debug=True, threaded=True)


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


