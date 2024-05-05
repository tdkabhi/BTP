import eventlet
eventlet.monkey_patch()

from flask import Flask, render_template, request, redirect, jsonify
from flask_socketio import SocketIO
from codecarbon import EmissionsTracker
from ultralytics import YOLO
import os
import psutil
import torch
import time
from concurrent.futures import ThreadPoolExecutor
from flask import Flask, render_template, url_for, redirect
from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin, login_user, LoginManager, login_required, logout_user, current_user
from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, SubmitField
from wtforms.validators import InputRequired, Length, ValidationError
from flask_bcrypt import Bcrypt
from zipfile import ZipFile


app = Flask(__name__)
db = SQLAlchemy(app)
bcrypt = Bcrypt(app)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///database.db'
app.config['SECRET_KEY'] = 'thisisasecretkey'


login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'zip'}
# Ensure the upload folder exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))


class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(20), nullable=False, unique=True)
    password = db.Column(db.String(80), nullable=False)


class RegisterForm(FlaskForm):
    username = StringField(validators=[
                           InputRequired(), Length(min=4, max=20)], render_kw={"placeholder": "Username"})

    password = PasswordField(validators=[
                             InputRequired(), Length(min=8, max=20)], render_kw={"placeholder": "Password"})

    submit = SubmitField('Register')

    def validate_username(self, username):
        existing_user_username = User.query.filter_by(
            username=username.data).first()
        if existing_user_username:
            raise ValidationError(
                'That username already exists. Please choose a different one.')


class LoginForm(FlaskForm):
    username = StringField(validators=[
                           InputRequired(), Length(min=4, max=20)], render_kw={"placeholder": "Username"})

    password = PasswordField(validators=[
                             InputRequired(), Length(min=8, max=20)], render_kw={"placeholder": "Password"})

    submit = SubmitField('Login')


@app.route('/')
def home():
    return render_template('home.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    form = LoginForm()
    if form.validate_on_submit():
        user = User.query.filter_by(username=form.username.data).first()
        if user:
            if bcrypt.check_password_hash(user.password, form.password.data):
                login_user(user)
                return redirect(url_for('index'))
    return render_template('login.html', form=form)



@app.route('/logout', methods=['GET', 'POST'])
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))


@ app.route('/register', methods=['GET', 'POST'])
def register():
    form = RegisterForm()

    if form.validate_on_submit():
        hashed_password = bcrypt.generate_password_hash(form.password.data)
        new_user = User(username=form.username.data, password=hashed_password)
        db.session.add(new_user)
        db.session.commit()
        return redirect(url_for('login'))

    return render_template('register.html', form=form)

socketio = SocketIO(app, async_mode='eventlet', cors_allowed_origins="*")

models_config = {
    'yolov8n': 'yolov8n.pt',
    'yolov8s': 'yolov8s.pt',
    'yolov8m': 'yolov8m.pt',
    'yolov8l': 'yolov8l.pt',
    'yolov8x': 'yolov8x.pt',
}

models = {name: YOLO(path) for name, path in models_config.items()}
active_models = set(models.keys())  # By default, all models are active
image_folder = 'train1'
cumulative_emissions = {name: 0 for name in models_config.keys()}  # Initialize cumulative emissions

@app.route('/index', methods=['GET'])
@login_required
def index():
    return render_template('Main.html')

@app.route('/charts.html', methods=['GET'])
@login_required
def charts():
    return render_template('charts.html')

@app.route('/dashboard', methods=['GET'])
def dashboard():
    return render_template('Main2.html')

@app.route('/set-models', methods=['POST'])
def set_models():
    selected_models = request.form.getlist('models')
    global active_models
    active_models = set(selected_models)
    return redirect('/dashboard')


def get_system_metrics():
    while True:
        cpu_usage = psutil.cpu_percent(interval=None)
        gpu_usage = get_gpu_memory()
        socketio.emit('system_metrics', {'cpu_usage': cpu_usage, 'gpu_usage': gpu_usage})
        time.sleep(1)

def get_gpu_memory():
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        total_memory = torch.cuda.get_device_properties(0).total_memory
        memory_allocated = torch.cuda.memory_allocated(0)
        return memory_allocated / total_memory * 100
    return 0

def process_image(model_name, model, image_path):
    if model_name not in active_models:
        return None  # Skip processing if model is not active
    tracker = EmissionsTracker()
    tracker.start()
    if torch.cuda.is_available():
        model.to('cuda')
    results = model(image_path)
    emissions = tracker.stop()
    cumulative_emissions[model_name] += emissions
    carbon_tax = cumulative_emissions[model_name] * 1500 / 1000  # Carbon tax in INR per metric ton of CO2
    return {
        'model': model_name,
        'emissions': emissions,
        'cumulative_emissions': cumulative_emissions[model_name],
        'carbon_tax': carbon_tax
    }

def emit_model_emissions():
    with ThreadPoolExecutor(max_workers=len(models)) as executor:
        while True:
            images = [os.path.join(image_folder, img) for img in os.listdir(image_folder) if img.lower().endswith(('.png', '.jpg', '.jpeg'))]
            if not images:
                print("No images found in the folder. Exiting.")
                break
            for image_path in images:
                futures = {model_name: executor.submit(process_image, model_name, model, image_path) for model_name, model in models.items() if model_name in active_models}
                results = [future.result() for future in futures.values() if future.result() is not None]
                if results:
                    socketio.emit('model_emissions', {'image': os.path.basename(image_path), 'results': results})
            time.sleep(10)  # Adjust as needed

@socketio.on('connect')
def on_connect():
    print('Client connected, starting emissions and system metrics emission...')
    eventlet.spawn(emit_model_emissions)
    eventlet.spawn(get_system_metrics)

if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=5000)
