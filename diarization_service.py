#!/usr/bin/env python3
"""
Flask сервис для диаризации аудио
Запускается как отдельный микросервис с авторизацией по Bearer token
"""

import os
import json
import tempfile
import torch
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

# Определение устройства для вычислений
def get_device():
    """Определяет наилучшее доступное устройство для вычислений"""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"🚀 CUDA available: {gpu_name} ({gpu_memory:.1f} GB)")
        
        # Проверяем доступную память
        torch.cuda.empty_cache()
        free_memory = torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated(0)
        free_memory_gb = free_memory / (1024**3)
        print(f"💾 GPU memory available: {free_memory_gb:.1f} GB")
        
        if free_memory_gb < 2.0:
            print("⚠️  Warning: Less than 2GB GPU memory available, performance may be limited")
        
        return device
    else:
        print("⚡ CUDA not available, using CPU")
        return torch.device('cpu')

DEVICE = get_device()


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
            pipeline = pipeline.to(DEVICE)
            print(f"✓ Model loaded from local path and moved to {DEVICE}")
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
            pipeline = pipeline.to(DEVICE)
            print(f"✓ Model loaded from HuggingFace cache and moved to {DEVICE}")
        
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
        
        # Информация о CUDA
        cuda_info = {
            'available': torch.cuda.is_available(),
            'current_device': str(DEVICE),
        }
        
        if torch.cuda.is_available():
            cuda_info.update({
                'device_count': torch.cuda.device_count(),
                'device_name': torch.cuda.get_device_name(0),
                'memory_total_gb': round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 2),
                'memory_allocated_gb': round(torch.cuda.memory_allocated(0) / (1024**3), 2),
                'memory_cached_gb': round(torch.cuda.memory_reserved(0) / (1024**3), 2)
            })
        
        return jsonify({
            'status': 'healthy',
            'model': model_status,
            'device': cuda_info,
            'torch_version': torch.__version__
        }), 200
    except Exception as e:
        return jsonify({
            'status': 'unhealthy',
            'error': str(e)
        }), 500


@app.route('/system', methods=['GET'])
def system_info():
    """Детальная информация о системе и CUDA"""
    try:
        system_info = {
            'python_version': f"{os.sys.version_info.major}.{os.sys.version_info.minor}.{os.sys.version_info.micro}",
            'torch_version': torch.__version__,
            'device': str(DEVICE),
            'cuda': {
                'available': torch.cuda.is_available(),
                'version': torch.version.cuda if torch.cuda.is_available() else None,
            }
        }
        
        if torch.cuda.is_available():
            system_info['cuda'].update({
                'device_count': torch.cuda.device_count(),
                'current_device': torch.cuda.current_device(),
                'devices': []
            })
            
            for i in range(torch.cuda.device_count()):
                device_props = torch.cuda.get_device_properties(i)
                device_info = {
                    'id': i,
                    'name': device_props.name,
                    'major': device_props.major,
                    'minor': device_props.minor,
                    'total_memory_gb': round(device_props.total_memory / (1024**3), 2),
                    'multiprocessor_count': device_props.multi_processor_count
                }
                
                # Только для текущего устройства получаем использование памяти
                if i == torch.cuda.current_device():
                    device_info.update({
                        'memory_allocated_gb': round(torch.cuda.memory_allocated(i) / (1024**3), 2),
                        'memory_cached_gb': round(torch.cuda.memory_reserved(i) / (1024**3), 2),
                        'memory_free_gb': round((device_props.total_memory - torch.cuda.memory_reserved(i)) / (1024**3), 2)
                    })
                
                system_info['cuda']['devices'].append(device_info)
        
        return jsonify(system_info), 200
        
    except Exception as e:
        return jsonify({
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
        import time
        print(f"🎵 Processing file: {file.filename} ({file_size / 1024:.2f} KB) on {DEVICE}")
        
        # Очищаем GPU кэш перед обработкой
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        start_time = time.time()
        model = load_pipeline()
        
        # Засекаем время диаризации
        diarization_start = time.time()
        diarization = model(temp_file)
        diarization_time = time.time() - diarization_start
        
        # Формирование результата
        result = []
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            result.append({
                "start": float(turn.start),
                "end": float(turn.end),
                "speaker": speaker
            })
        
        total_time = time.time() - start_time
        print(f"✓ Diarization completed: {len(result)} segments in {diarization_time:.2f}s (total: {total_time:.2f}s)")
        
        return jsonify({
            'success': True,
            'segments': result,
            'total_segments': len(result),
            'processing_time_seconds': round(diarization_time, 2),
            'total_time_seconds': round(total_time, 2),
            'device': str(DEVICE)
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
