from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify, Response
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime
import secrets
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import io
import base64
import os
import json
import threading
import time
import logging
from dotenv import load_dotenv

import mediapipe as mp

load_dotenv()

app = Flask(__name__)
app.secret_key = secrets.token_hex(32)

# Database Configuration
app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv("DB_URI")
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['UPLOAD_FOLDER'] = 'static/uploads'
db = SQLAlchemy(app)

if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

if not os.path.exists('static'):
    os.makedirs('static')

# User Model
class User(db.Model):
    __tablename__ = 'user'
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    username = db.Column(db.String(50), nullable=False, unique=True)
    email = db.Column(db.String(100), nullable=False, unique=True)
    password = db.Column(db.String(1023), nullable=False)
    profile_pic = db.Column(db.String(120), default='defaultprofile.jpg')

class SessionHistory(db.Model):
    __tablename__ = 'session_history'
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    login_time = db.Column(db.DateTime, default=db.func.now())
    logout_time = db.Column(db.DateTime, nullable=True)
    user = db.relationship('User', backref=db.backref('sessions', lazy=True))

class EmotionAnalysis(db.Model):
    __tablename__ = 'emotion_analysis'
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    timestamp = db.Column(db.DateTime, default=db.func.now())
    emotion = db.Column(db.String(50), nullable=False)
    confidence = db.Column(db.Float, nullable=False)
    user = db.relationship('User', backref=db.backref('emotions', lazy=True))

# Load Pretrained ML Model
try:
    model = load_model('emotion_model.h5')
    print(model.summary())
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

import threading
mp_lock = threading.Lock()

def detect_landmarks(image):
    mp_face_mesh = mp.solutions.face_mesh
    with mp_lock:
        with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1) as face_mesh:
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb_image)
            if results.multi_face_landmarks:
                h, w, _ = image.shape
                landmarks = []
                for lm in results.multi_face_landmarks[0].landmark:
                    x, y = int(lm.x * w), int(lm.y * h)
                    landmarks.append([x, y])
                return np.array(landmarks)
            return None


def predict_emotion(image):
    if model is None:
        return {'error': 'Model not loaded'}
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb_image, (48, 48)) / 255.0
    prediction = model.predict(np.expand_dims(resized, axis=0))[0]
    emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
    return {emotion_labels[i]: float(prediction[i]) for i in range(len(emotion_labels))}

class EmotionTracker:
    def __init__(self):
        self.emotions = []
        self.lock = threading.Lock()
    def add_emotion(self, emotion, confidence):
        with self.lock:
            self.emotions.append((emotion, confidence, datetime.now()))
    def get_results(self):
        with self.lock:
            if not self.emotions:
                return "neutral", {"neutral": 1.0}
            emotion_counts = {}
            for emotion, _, _ in self.emotions:
                emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
            total = len(self.emotions)
            emotion_percentages = {emotion: count / total for emotion, count in emotion_counts.items()}
            dominant_emotion = max(emotion_percentages, key=emotion_percentages.get)
            return dominant_emotion, emotion_percentages
    def clear(self):
        with self.lock:
            self.emotions = []

emotion_trackers = {}

@app.route('/')
def default():
    return redirect(url_for('about'))

@app.route('/about')
def about():
    if 'user_id' not in session:
        return render_template('about.html')
    return redirect(url_for('login'))

@app.route('/index')
def index():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    return render_template('index.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        user = User.query.filter_by(email=email).first()
        if user and check_password_hash(user.password, password):
            session['user_id'] = user.id
            session['username'] = user.username
            new_session = SessionHistory(user_id=user.id)
            db.session.add(new_session)
            db.session.commit()
            return redirect(url_for('index'))
        flash('Invalid email or password', 'danger')
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['name']
        email = request.form['email']
        password = request.form['password']
        confirm_password = request.form['confirm-password']
        if password != confirm_password:
            flash('Passwords do not match', 'danger')
            return redirect(url_for('register'))
        existing_user = User.query.filter((User.email == email) | (User.username == username)).first()
        if existing_user:
            flash('Email or Username already exists', 'danger')
            return redirect(url_for('register'))
        new_user = User(username=username, email=email, password=generate_password_hash(password))
        db.session.add(new_user)
        db.session.commit()
        flash('Registration successful! Please log in.', 'success')
        return redirect(url_for('login'))
    return render_template('register.html')

@app.route('/terms')
def terms():
    return render_template('terms.html')

@app.route('/privacy')
def privacy():
    return render_template('privacy.html')

@app.route('/logout')
def logout():
    if 'user_id' in session:
        user_id = session['user_id']
        last_session = SessionHistory.query.filter_by(user_id=user_id, logout_time=None).order_by(
            SessionHistory.login_time.desc()).first()
        if last_session:
            last_session.logout_time = datetime.now()
            db.session.commit()
        session.clear()
    return redirect(url_for('login'))

@app.route('/analyze', methods=['POST'])
def analyze():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    file = request.files['image']
    if not file:
        return redirect(url_for('camera'))
    image = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
    landmarks = detect_landmarks(image)
    if landmarks is None:
        return redirect(url_for('camera'))
    emotion = predict_emotion(image)
    new_analysis = EmotionAnalysis(user_id=session['user_id'], emotion=max(emotion, key=emotion.get),
                                   confidence=max(emotion.values()))
    db.session.add(new_analysis)
    db.session.commit()
    return render_template('analysis.html', emotion=emotion, landmarks=landmarks.tolist())

@app.route('/history')
def history():
    if 'user_id' not in session:
        flash('Please log in first.', 'warning')
        return redirect(url_for('login'))
    user_id = session['user_id']
    session_records = SessionHistory.query.filter_by(user_id=user_id).order_by(SessionHistory.login_time.desc()).all()
    sessions_with_emotions = []
    for session_record in session_records:
        emotions_in_session = EmotionAnalysis.query.filter_by(user_id=user_id).filter(
            EmotionAnalysis.timestamp >= session_record.login_time,
            EmotionAnalysis.timestamp <= (session_record.logout_time if session_record.logout_time else datetime.now())
        ).all()
        predominant_emotion = "None" if not emotions_in_session else max((e.emotion for e in emotions_in_session),
                                                                       key=lambda x: sum(1 for e in emotions_in_session if e.emotion == x))
        sessions_with_emotions.append({
            'session': session_record,
            'predominant_emotion': predominant_emotion,
            'emotion_count': len(emotions_in_session)
        })
    emotion_records = EmotionAnalysis.query.filter_by(user_id=user_id).order_by(EmotionAnalysis.timestamp.desc()).all()
    return render_template('history.html',
                          sessions_with_emotions=sessions_with_emotions,
                          emotion_records=emotion_records)

@app.route('/session_emotions/<int:session_id>')
def session_emotions(session_id):
    if 'user_id' not in session:
        return jsonify({'error': 'Not logged in'}), 401
    user_id = session['user_id']
    session_record = SessionHistory.query.filter_by(id=session_id, user_id=user_id).first()
    if not session_record:
        return jsonify({'error': 'Session not found'}), 404
    emotions_in_session = EmotionAnalysis.query.filter_by(user_id=user_id).filter(
        EmotionAnalysis.timestamp >= session_record.login_time,
        EmotionAnalysis.timestamp <= session_record.logout_time if session_record.logout_time else datetime.now()
    ).all()
    emotion_counts = {}
    for emotion_record in emotions_in_session:
        emotion_counts[emotion_record.emotion] = emotion_counts.get(emotion_record.emotion, 0) + 1
    total = len(emotions_in_session) if emotions_in_session else 1
    emotion_percentages = {emotion: (count / total) * 100 for emotion, count in emotion_counts.items()}
    return jsonify({
        'session_id': session_id,
        'emotions': emotion_counts,
        'percentages': emotion_percentages,
        'total_records': total
    })

@app.route('/get_profile', methods=['GET'])
def get_profile():
    if 'user_id' not in session:
        return jsonify({'success': False, 'message': 'Not logged in'}), 401
    user = User.query.get(session['user_id'])
    if not user:
        return jsonify({'success': False, 'message': 'User not found'}), 404
    first_login = SessionHistory.query.filter_by(user_id=user.id).order_by(SessionHistory.login_time.asc()).first()
    days_active = (datetime.now() - first_login.login_time).days if first_login else 0
    total_activities = SessionHistory.query.filter_by(user_id=user.id).count()
    return jsonify({
        'success': True,
        'username': user.username,
        'email': user.email,
        'profile_pic': url_for('static', filename=user.profile_pic),
        'days_active': days_active,
        'total_activities': total_activities
    })

@app.route('/show_profile', methods=['GET'])
def show_profile():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    return render_template('profile.html')

@app.route('/update_profile', methods=['POST'])
def update_profile():
    if 'user_id' not in session:
        return jsonify({'success': False, 'message': 'Not logged in'}), 401
    user = User.query.get(session['user_id'])
    if not user:
        return jsonify({'success': False, 'message': 'User not found'}), 404

    username = request.form.get('username')
    email = request.form.get('email')
    password = request.form.get('password')

    existing_user = User.query.filter(
        ((User.username == username) | (User.email == email)) & (User.id != user.id)
    ).first()
    if existing_user:
        return jsonify({'success': False, 'message': 'Username or email already exists'}), 400

    if username:
        user.username = username
    if email:
        user.email = email
    if password:
        user.password = generate_password_hash(password)

    if 'profile-pic-upload' in request.files:
        file = request.files['profile-pic-upload']
        if file and file.filename:
            from werkzeug.utils import secure_filename
            existing_files = [f for f in os.listdir(app.config['UPLOAD_FOLDER']) if f.startswith(f"{user.username}_pic")]
            max_version = 0
            for f in existing_files:
                try:
                    version = int(f.split('_pic')[-1].split('.')[0])
                    max_version = max(max_version, version)
                except ValueError:
                    continue
            new_version = max_version + 1
            filename = f"{user.username}_pic{new_version}.jpg"
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            user.profile_pic = f"uploads/{filename}"

    db.session.commit()
    return jsonify({
        'success': True,
        'username': user.username,
        'email': user.email,
        'profile_pic': url_for('static', filename=user.profile_pic)
    })

@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/process_image', methods=['POST'])
def process_image():
    if 'user_id' not in session:
        return jsonify({'error': 'Not logged in'}), 401
    file = request.files['image']
    if not file:
        return jsonify({'error': 'No image uploaded'}), 400
    image = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        return jsonify({'error': 'Invalid image'}), 400
    landmarks = detect_landmarks(image)
    if landmarks is None:
        return jsonify({'error': 'No face detected'}), 400
    scores = predict_emotion(image)
    dominant_emotion = max(scores, key=scores.get)
    new_analysis = EmotionAnalysis(user_id=session['user_id'], emotion=dominant_emotion,
                                   confidence=max(scores.values()))
    db.session.add(new_analysis)
    db.session.commit()
    return jsonify({
        'emotion': dominant_emotion,
        'scores': json.dumps(scores)
    })

@app.route('/camera')
def camera():
    return render_template('camera.html')

@app.route('/upload')
def upload():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    return render_template('upload.html')

@app.route('/videoscan')
def videoscan():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    return render_template('videoscan.html')

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

@app.route('/process_upload', methods=['POST'])
def process_upload():
    if 'user_id' not in session:
        logger.debug("User not logged in, redirecting to login")
        return redirect(url_for('login'))
    if 'image' not in request.files:
        logger.debug("No file part in request")
        flash('No file part', 'danger')
        return redirect(url_for('upload'))
    file = request.files['image']
    if file.filename == '':
        logger.debug("No file selected")
        flash('No selected file', 'danger')
        return redirect(url_for('upload'))
    logger.debug(f"Processing file: {file.filename}")
    try:
        image = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            logger.debug("Failed to read image")
            flash('Failed to read image. Please upload a valid image file.', 'danger')
            return redirect(url_for('upload'))
        logger.debug("Image read successfully")
        landmarks = detect_landmarks(image)
        if landmarks is None:
            logger.debug("No face detected in image")
            flash('No face detected in the image', 'warning')
            return redirect(url_for('upload'))
        logger.debug("Facial landmarks detected")
        scores = predict_emotion(image)
        if 'error' in scores:
            logger.debug(f"Emotion prediction error: {scores['error']}")
            flash(f'Error in emotion prediction: {scores["error"]}', 'danger')
            return redirect(url_for('upload'))
        logger.debug(f"Emotion scores: {scores}")
        dominant_emotion = max(scores, key=scores.get)
        logger.debug(f"Dominant emotion: {dominant_emotion}")
        new_analysis = EmotionAnalysis(
            user_id=session['user_id'],
            emotion=dominant_emotion,
            confidence=max(scores.values())
        )
        db.session.add(new_analysis)
        db.session.commit()
        logger.debug("Analysis saved to database")
        return redirect(url_for('result', emotion=dominant_emotion, scores=json.dumps(scores)))
    except Exception as e:
        logger.error(f"Exception occurred: {str(e)}")
        flash(f'Error processing image: {str(e)}', 'danger')
        return redirect(url_for('upload'))

@app.route('/video_feed')
def video_feed():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    user_id = session['user_id']
    if user_id not in emotion_trackers:
        emotion_trackers[user_id] = EmotionTracker()
    def generate():
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Could not open camera.")
            return
        last_emotion_time = time.time()
        emotion_interval = 1.0
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                current_time = time.time()
                if current_time - last_emotion_time > emotion_interval:
                    emotion_scores = predict_emotion(frame)
                    if 'error' not in emotion_scores:
                        dominant_emotion = max(emotion_scores, key=emotion_scores.get)
                        confidence = emotion_scores[dominant_emotion]
                        emotion_trackers[user_id].add_emotion(dominant_emotion, confidence)
                        if len(emotion_trackers[user_id].emotions) % 5 == 0:
                            with app.app_context():
                                new_analysis = EmotionAnalysis(
                                    user_id=user_id,
                                    emotion=dominant_emotion,
                                    confidence=confidence
                                )
                                db.session.add(new_analysis)
                                db.session.commit()
                        last_emotion_time = current_time
                    cv2.putText(
                        frame,
                        f"Emotion: {dominant_emotion} ({confidence:.2f})",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 255, 0),
                        2
                    )
                # NO landmark drawing code here!
                ret, jpeg = cv2.imencode('.jpg', frame)
                frame_bytes = jpeg.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        except Exception as e:
            print(f"Error in video stream: {e}")
        finally:
            cap.release()
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/end_session')
def end_session():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    user_id = session['user_id']
    if user_id in emotion_trackers:
        dominant_emotion, scores = emotion_trackers[user_id].get_results()
        emotions = list(scores.keys())
        values = list(scores.values())
        color_map = {'angry': 'red', 'disgust': 'green', 'fear': 'purple', 'happy': 'yellow',
                     'neutral': 'gray', 'sad': 'blue', 'surprise': 'orange'}
        colors = [color_map.get(e, 'gray') for e in emotions]
        plt.figure(figsize=(6, 6))
        plt.pie(values, labels=emotions, colors=colors, autopct='%1.1f%%', startangle=90)
        plt.axis('equal')
        img = io.BytesIO()
        plt.savefig(img, format='png', bbox_inches='tight')
        img.seek(0)
        pie_chart_base64 = base64.b64encode(img.getvalue()).decode()
        new_analysis = EmotionAnalysis(
            user_id=user_id,
            emotion=dominant_emotion,
            confidence=max(scores.values())
        )
        db.session.add(new_analysis)
        db.session.commit()
        emotion_trackers[user_id].clear()
        suggestions_dict = {
            'happy': ["Keep spreading positivity—share your joy with others! <a href='https://www.youtube.com/watch?v=X1GNc70-584' target='_blank'>Watch this podcast</a>", "Try a new hobby to maintain your high spirits.", "Reflect on what’s making you happy and how to sustain it."],
            'sad': ["Take a moment to relax—maybe watch a comforting movie. <a href='https://www.youtube.com/watch?v=h-3bixYKBFg' target='_blank'>Watch this podcast</a>", "Talk to a friend or loved one for support.", "Consider journaling your feelings to process them."],
            'angry': ["Take deep breaths or step away to cool off. <a href='https://www.youtube.com/watch?v=J8LMUkuxkbU' target='_blank'>Watch this podcast</a>", "Engage in physical activity like a quick walk to release tension.", "Write down what’s bothering you to gain perspective."],
            'fear': ["Focus on what you can control to ease your worries. <a href='https://www.youtube.com/watch?v=q6VRPyX1qHg' target='_blank'>Watch this podcast</a>", "Practice a grounding exercise, like counting to 10 slowly.", "Talk to someone you trust about what’s on your mind."],
            'disgust': ["Shift your focus to something you enjoy to reset. <a href='https://www.youtube.com/watch?v=oUTEJGEGGPE' target='_blank'>Watch this podcast</a>", "Take a break from whatever’s bothering you.", "Clean or organize your space for a fresh start."],
            'neutral': ["Try something new to spark some excitement. <a href='https://www.youtube.com/watch?v=x4JzmIZAPxs' target='_blank'>Watch this podcast</a>", "Set a small goal for the day to feel accomplished. <a href='https://www.youtube.com/watch?v=tnVMB0P-FYM' target='_blank'>Watch this podcast</a>", "Take a moment to appreciate the calm. <a href='https://www.youtube.com/watch?v=QnEVmSFJB9g' target='_blank'>Watch this podcast</a>"],
            'surprise': ["Embrace the unexpected—see where it takes you! <a href='https://www.youtube.com/watch?v=2kdvTv7jg1s' target='_blank'>Watch this podcast</a>", "Pause and assess what surprised you to stay grounded.", "Share your reaction with someone for a fun conversation."]
        }
        suggestions = suggestions_dict.get(dominant_emotion.lower(), ["Take a moment to reflect on your current state."])
        return render_template('result.html',
                             emotion=dominant_emotion,
                             scores=scores,
                             pie_chart_data=pie_chart_base64,
                             suggestions=suggestions)
    flash('No emotion data recorded', 'warning')
    return redirect(url_for('videoscan'))

@app.route('/current_emotion_stats')
def current_emotion_stats():
    if 'user_id' not in session:
        return jsonify({'error': 'Not logged in'}), 401
    user_id = session['user_id']
    if user_id in emotion_trackers:
        dominant_emotion, scores = emotion_trackers[user_id].get_results()
        return jsonify({
            'dominant_emotion': dominant_emotion,
            'scores': scores
        })
    return jsonify({'error': 'No emotion data available'}), 404

@app.route('/result')
def result():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    emotion = request.args.get('emotion', 'Unknown')
    scores_str = request.args.get('scores', '{}')
    try:
        scores = json.loads(scores_str.replace("'", '"'))
    except json.JSONDecodeError:
        scores = {}
    emotions = list(scores.keys())
    values = list(scores.values())
    color_map = {'angry': 'red', 'disgust': 'green', 'fear': 'purple', 'happy': 'yellow',
                 'neutral': 'gray', 'sad': 'blue', 'surprise': 'orange'}
    colors = [color_map.get(e, 'gray') for e in emotions]
    plt.figure(figsize=(6, 6))
    plt.pie(values, labels=emotions, colors=colors, autopct='%1.1f%%', startangle=90)
    plt.axis('equal')
    img = io.BytesIO()
    plt.savefig(img, format='png', bbox_inches='tight')
    img.seek(0)
    pie_chart_base64 = base64.b64encode(img.getvalue()).decode()
    suggestions_dict = {
        'happy': ["Keep spreading positivity—share your joy with others! <a href='https://www.youtube.com/watch?v=X1GNc70-584' target='_blank'>Watch this podcast</a>", "Try a new hobby to maintain your high spirits.", "Reflect on what’s making you happy and how to sustain it."],
        'sad': ["Take a moment to relax—maybe watch a comforting movie. <a href='https://www.youtube.com/watch?v=h-3bixYKBFg' target='_blank'>Watch this podcast</a>", "Talk to a friend or loved one for support.", "Consider journaling your feelings to process them."],
        'angry': ["Take deep breaths or step away to cool off. <a href='https://www.youtube.com/watch?v=J8LMUkuxkbU' target='_blank'>Watch this podcast</a>", "Engage in physical activity like a quick walk to release tension.", "Write down what’s bothering you to gain perspective."],
        'fear': ["Focus on what you can control to ease your worries. <a href='https://www.youtube.com/watch?v=q6VRPyX1qHg' target='_blank'>Watch this podcast</a>", "Practice a grounding exercise, like counting to 10 slowly.", "Talk to someone you trust about what’s on your mind."],
        'disgust': ["Shift your focus to something you enjoy to reset. <a href='https://www.youtube.com/watch?v=oUTEJGEGGPE' target='_blank'>Watch this podcast</a>", "Take a break from whatever’s bothering you.", "Clean or organize your space for a fresh start."],
        'neutral': ["Try something new to spark some excitement. <a href='https://www.youtube.com/watch?v=x4JzmIZAPxs' target='_blank'>Watch this podcast</a>", "Set a small goal for the day to feel accomplished. <a href='https://www.youtube.com/watch?v=tnVMB0P-FYM' target='_blank'>Watch this podcast</a>", "Take a moment to appreciate the calm. <a href='https://www.youtube.com/watch?v=QnEVmSFJB9g' target='_blank'>Watch this podcast</a>"],
        'surprise': ["Embrace the unexpected—see where it takes you! <a href='https://www.youtube.com/watch?v=2kdvTv7jg1s' target='_blank'>Watch this podcast</a>", "Pause and assess what surprised you to stay grounded.", "Share your reaction with someone for a fun conversation."]
    }
    suggestions = suggestions_dict.get(emotion.lower(), ["Take a moment to reflect on your current state."])
    return render_template('result.html',
                         emotion=emotion,
                         scores=scores,
                         pie_chart_data=pie_chart_base64,
                         suggestions=suggestions)

with app.app_context():
    db.create_all()

if __name__ == '__main__':
    app.run(debug=True)
