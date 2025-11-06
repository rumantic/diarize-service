#!/usr/bin/env python3
"""
Flask сервис для диаризации аудио
Запускается как отдельный микросервис с авторизацией по Bearer token
"""

import os
import json
import tempfile
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
from functools import wraps
from pyannote.audio import Pipeline

# Загрузка переменных окружения из .env файла
from dotenv import load_dotenv
load_dotenv()

# Настройка прокси из переменных окружения
def setup_proxy():
    """Настройка прокси для загрузки моделей"""
    http_proxy = os.getenv('HTTP_PROXY') or os.getenv('http_proxy')
    https_proxy = os.getenv('HTTPS_PROXY') or os.getenv('https_proxy')
    no_proxy = os.getenv('NO_PROXY') or os.getenv('no_proxy')
    
    proxies = {}
    
    if http_proxy or https_proxy:
        print("Proxy configuration detected:")
        if http_proxy:
            print(f"  HTTP:  {http_proxy}")
            os.environ['http_proxy'] = http_proxy
            os.environ['HTTP_PROXY'] = http_proxy
            proxies['http'] = http_proxy
        if https_proxy:
            print(f"  HTTPS: {https_proxy}")
            os.environ['https_proxy'] = https_proxy
            os.environ['HTTPS_PROXY'] = https_proxy
            proxies['https'] = https_proxy
        if no_proxy:
            os.environ['no_proxy'] = no_proxy
            os.environ['NO_PROXY'] = no_proxy
        
        # Настраиваем прокси для urllib
        try:
            import urllib.request
            proxy_handler = urllib.request.ProxyHandler(proxies)
            opener = urllib.request.build_opener(proxy_handler)
            urllib.request.install_opener(opener)
        except Exception as e:
            print(f"⚠ Warning: Could not configure urllib proxy: {e}")
        
        # Настраиваем прокси для huggingface_hub
        try:
            from huggingface_hub import configure_http_backend
            import requests
            
            session = requests.Session()
            session.proxies.update(proxies)
            session.verify = True
            
            configure_http_backend(backend_factory=lambda: session)
            print("✓ HuggingFace Hub configured with proxy")
        except Exception as e:
            print(f"⚠ Warning: Could not configure HF Hub proxy: {e}")
    
    return proxies

setup_proxy()

app = Flask(__name__)

# Конфигурация
BEARER_TOKEN = os.getenv('DIARIZATION_TOKEN', 'your-secret-token-here')
MODEL_PATH = os.getenv('PYANNOTE_MODEL_PATH', './models/pyannote-speaker-diarization-3.1')
MAX_FILE_SIZE = 100 * 1024 * 1024  # 100 MB
ALLOWED_EXTENSIONS = {'mp3', 'wav', 'ogg', 'm4a', 'flac', 'wma'}

# Глобальная переменная для хранения загруженной модели
pipeline = None


def load_pipeline():
    """Загрузка модели при старте сервиса"""
    global pipeline
    
    if pipeline is not None:
        return pipeline
    
    model_path = MODEL_PATH
    print(f"Loading diarization model...")
    print(f"Local path configured: {model_path}")
    
    try:
        # Сначала пробуем загрузить из локального пути
        if model_path and os.path.exists(model_path):
            print(f"Loading from local path: {model_path}")
            
            # Конвертируем в абсолютный путь и заменяем \ на / для pyannote
            abs_model_path = os.path.abspath(model_path).replace('\\', '/')
            print(f"Absolute path: {abs_model_path}")
            
            # Для Windows путей используем file:// URI схему
            if os.name == 'nt':  # Windows
                # Проверяем, что все файлы на месте
                config_file = os.path.join(model_path, 'config.yaml')
                model_file = os.path.join(model_path, 'pytorch_model.bin')
                
                if not os.path.exists(config_file):
                    raise Exception(f"config.yaml not found in {model_path}")
                if not os.path.exists(model_file):
                    raise Exception(f"pytorch_model.bin not found in {model_path}")
                
                print("Found config.yaml and pytorch_model.bin")
                
            pipeline = Pipeline.from_pretrained(abs_model_path)
            print("✓ Model loaded from local path")
        else:
            # Загружаем из HuggingFace (используется кэш если модель уже скачана)
            hf_token = os.getenv('HF_TOKEN')
            
            print("Loading from HuggingFace (will use cache if available)...")
            
            # Устанавливаем токен через переменную окружения
            if hf_token:
                os.environ['HF_TOKEN'] = hf_token
                print("Using HF_TOKEN for authentication")
            
            # Увеличиваем таймауты для медленного интернета/прокси
            os.environ['HF_HUB_DOWNLOAD_TIMEOUT'] = '600'
            
            # Загружаем без явной передачи токена - будет использован из env или кэша
            pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1")
            print("✓ Model loaded from HuggingFace cache")
        
        return pipeline
    
    except Exception as e:
        print(f"✗ Error loading model: {str(e)}")
        raise


def require_token(f):
    """Декоратор для проверки Bearer token"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        auth_header = request.headers.get('Authorization')
        
        if not auth_header:
            return jsonify({'error': 'No authorization header'}), 401
        
        try:
            scheme, token = auth_header.split()
            if scheme.lower() != 'bearer':
                return jsonify({'error': 'Invalid authorization scheme'}), 401
            
            if token != BEARER_TOKEN:
                return jsonify({'error': 'Invalid token'}), 401
        
        except ValueError:
            return jsonify({'error': 'Invalid authorization header format'}), 401
        
        return f(*args, **kwargs)
    
    return decorated_function


def allowed_file(filename):
    """Проверка расширения файла"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/health', methods=['GET'])
def health():
    """Проверка здоровья сервиса"""
    try:
        model_status = "loaded" if pipeline is not None else "not loaded"
        return jsonify({
            'status': 'healthy',
            'model': model_status
        }), 200
    except Exception as e:
        return jsonify({
            'status': 'unhealthy',
            'error': str(e)
        }), 500


@app.route('/diarize', methods=['POST'])
@require_token
def diarize():
    """
    Эндпоинт для диаризации аудио
    
    Принимает:
    - file: аудиофайл (multipart/form-data)
    
    Возвращает:
    - JSON с массивом сегментов (speaker, start, end)
    """
    
    # Проверка наличия файла
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'Empty filename'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': f'File type not allowed. Allowed: {", ".join(ALLOWED_EXTENSIONS)}'}), 400
    
    # Проверка размера файла
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)
    
    if file_size > MAX_FILE_SIZE:
        return jsonify({'error': f'File too large. Max size: {MAX_FILE_SIZE / 1024 / 1024} MB'}), 400
    
    # Сохранение во временный файл
    temp_file = None
    try:
        # Создаём временный файл
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as tmp:
            temp_file = tmp.name
            file.save(tmp.name)
        
        # Выполнение диаризации
        print(f"Processing file: {file.filename} ({file_size / 1024:.2f} KB)")
        
        model = load_pipeline()
        diarization = model(temp_file)
        
        # Формирование результата
        result = []
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            result.append({
                "start": float(turn.start),
                "end": float(turn.end),
                "speaker": speaker
            })
        
        print(f"✓ Diarization completed: {len(result)} segments")
        
        return jsonify({
            'success': True,
            'segments': result,
            'total_segments': len(result)
        }), 200
    
    except Exception as e:
        print(f"✗ Error during diarization: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500
    
    finally:
        # Удаление временного файла
        if temp_file and os.path.exists(temp_file):
            try:
                os.unlink(temp_file)
            except:
                pass


if __name__ == '__main__':
    # Загрузка модели при старте
    try:
        load_pipeline()
    except Exception as e:
        print(f"WARNING: Could not load model at startup: {str(e)}")
        print("Model will be loaded on first request")
    
    # Запуск сервера
    port = int(os.getenv('DIARIZATION_PORT', 5000))
    host = os.getenv('DIARIZATION_HOST', '0.0.0.0')
    
    print(f"\n{'='*60}")
    print(f"Diarization Service Starting")
    print(f"{'='*60}")
    print(f"Host: {host}")
    print(f"Port: {port}")
    print(f"Token: {BEARER_TOKEN[:10]}... (set via DIARIZATION_TOKEN)")
    print(f"Model: {MODEL_PATH}")
    print(f"{'='*60}\n")
    
    app.run(host=host, port=port, debug=False)
