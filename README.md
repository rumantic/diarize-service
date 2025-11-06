# Audio Diarization Service

Микросервис на Flask для диаризации аудио (определение говорящих) с использованием pyannote.audio.

## Описание

Сервис предоставляет REST API для обработки аудиофайлов и определения временных меток говорящих (speaker diarization). Использует модель `pyannote/speaker-diarization-3.1` от HuggingFace.

## Возможности

- 🎯 Определение говорящих в аудиофайлах
- 🔐 Аутентификация через Bearer Token
- 💾 Локальное кэширование модели (без зависимости от HuggingFace после загрузки)
- 🚀 REST API с JSON-ответами
- ✅ Health check endpoint
- 📊 Поддержка различных аудиоформатов (MP3, WAV, M4A и др.)

## Требования

- Python 3.8+
- FFmpeg (для обработки аудио)
- ~3GB свободного места для модели

## Установка

### 1. Установка зависимостей

```bash
pip install -r requirements.txt
```

### 2. Получение токена HuggingFace

1. Зарегистрируйтесь на [HuggingFace](https://huggingface.co/)
2. Создайте токен доступа в [Settings → Access Tokens](https://huggingface.co/settings/tokens)
3. Примите лицензионное соглашение модели [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1)

### 3. Загрузка модели

Выполните скрипт загрузки модели (требуется только один раз):

```bash
# Windows PowerShell
$env:HF_TOKEN = "your_huggingface_token_here"
python download_model.py

# Linux/Mac
export HF_TOKEN=your_huggingface_token_here
python download_model.py
```

Модель будет сохранена в директорию `./models/pyannote/`.

### 4. Настройка переменных окружения

Создайте файл `.env` или установите переменные окружения:

```bash
# Обязательные
DIARIZATION_TOKEN=your_secure_random_token_here
HF_TOKEN=your_huggingface_token  # Только для первого запуска

# Опциональные
PYANNOTE_MODEL_PATH=./models/pyannote  # Путь к локальной модели
DIARIZATION_PORT=5000                   # Порт сервиса
DIARIZATION_HOST=127.0.0.1             # Хост сервиса
```

## Запуск

### Разработка

```bash
python diarization_service.py
```

### Production (с Gunicorn)

```bash
gunicorn -w 4 -b 0.0.0.0:5000 --timeout 300 diarization_service:app
```

## API Endpoints

### Health Check

Проверка работоспособности сервиса.

```http
GET /health
```

**Ответ:**
```json
{
    "status": "healthy",
    "model_loaded": true
}
```

### Diarization

Диаризация аудиофайла.

```http
POST /diarize
Authorization: Bearer <your_token>
Content-Type: multipart/form-data

file: <audio_file>
```

**Параметры:**
- `file` - аудиофайл (до 100MB)

**Пример запроса (curl):**
```bash
curl -X POST http://localhost:5000/diarize \
  -H "Authorization: Bearer your_token_here" \
  -F "file=@/path/to/audio.mp3"
```

**Пример запроса (PowerShell):**
```powershell
$headers = @{
    "Authorization" = "Bearer your_token_here"
}
$file = @{
    file = Get-Item "C:\path\to\audio.mp3"
}
Invoke-RestMethod -Uri "http://localhost:5000/diarize" -Method Post -Headers $headers -Form $file
```

**Успешный ответ:**
```json
{
    "speakers": [
        {
            "speaker": "SPEAKER_00",
            "start": 0.5,
            "end": 3.2
        },
        {
            "speaker": "SPEAKER_01",
            "start": 3.5,
            "end": 7.8
        }
    ],
    "duration": 10.5
}
```

**Ошибки:**
```json
{
    "error": "No file uploaded"
}
```

## Интеграция с Laravel

Сервис интегрирован с Laravel приложением через `AudioRecognitionTask`.

### Конфигурация в Laravel

Добавьте в `.env`:

```env
DIARIZATION_SERVICE_URL=http://localhost:5000
DIARIZATION_SERVICE_TOKEN=your_secure_random_token_here
```

### Использование

Laravel автоматически использует сервис диаризации при обработке аудио через API:

```php
// AudioRecognitionTask автоматически вызывает сервис
$task = new AudioRecognitionTask($apiRequest);
$result = $task->handle();

// Результат содержит transcription + diarization
// $result['transcription'] - текст от OpenAI Whisper
// $result['diarization'] - говорящие с временными метками
```

### Fallback

Если сервис недоступен, Laravel автоматически использует локальный Python скрипт `diarize.py`.

## Production Deployment

### Systemd Service

Создайте файл `/etc/systemd/system/diarization.service`:

```ini
[Unit]
Description=Audio Diarization Service
After=network.target

[Service]
Type=simple
User=www-data
WorkingDirectory=/path/to/scripts
Environment="DIARIZATION_TOKEN=your_token"
Environment="PYANNOTE_MODEL_PATH=/path/to/models/pyannote"
ExecStart=/usr/bin/python3 /path/to/scripts/diarization_service.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

Запуск:
```bash
sudo systemctl enable diarization
sudo systemctl start diarization
sudo systemctl status diarization
```

### Nginx Reverse Proxy

```nginx
location /diarization/ {
    proxy_pass http://localhost:5000/;
    proxy_set_header Host $host;
    proxy_set_header X-Real-IP $remote_addr;
    proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    
    # Увеличенные таймауты для больших файлов
    proxy_read_timeout 300s;
    proxy_connect_timeout 300s;
    proxy_send_timeout 300s;
    
    # Увеличенный размер загружаемых файлов
    client_max_body_size 100M;
}
```

### Docker

```dockerfile
FROM python:3.10-slim

WORKDIR /app

# Установка FFmpeg
RUN apt-get update && apt-get install -y ffmpeg && rm -rf /var/lib/apt/lists/*

# Установка зависимостей
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Копирование файлов
COPY diarization_service.py .
COPY models/ ./models/

# Переменные окружения
ENV DIARIZATION_PORT=5000
ENV DIARIZATION_HOST=0.0.0.0

EXPOSE 5000

CMD ["python", "diarization_service.py"]
```

```bash
docker build -t diarization-service .
docker run -d -p 5000:5000 \
  -e DIARIZATION_TOKEN=your_token \
  -e PYANNOTE_MODEL_PATH=/app/models/pyannote \
  --name diarization \
  diarization-service
```

## Безопасность

⚠️ **Важно:**

- Используйте сильный случайный токен для `DIARIZATION_TOKEN`
- Не коммитьте токены в Git
- В production используйте HTTPS
- Ограничьте доступ к сервису через firewall
- Храните модель в безопасной директории с ограниченными правами

Генерация токена:
```bash
# Linux/Mac
openssl rand -hex 32

# Python
python -c "import secrets; print(secrets.token_hex(32))"

# PowerShell
[System.Convert]::ToBase64String([System.Security.Cryptography.RandomNumberGenerator]::GetBytes(32))
```

## Troubleshooting

### Ошибка "Model not found"

Убедитесь, что модель загружена:
```bash
python download_model.py
```

### Ошибка "FFmpeg not found"

Установите FFmpeg:
- **Windows**: `choco install ffmpeg` или скачайте с [ffmpeg.org](https://ffmpeg.org/)
- **Ubuntu/Debian**: `sudo apt-get install ffmpeg`
- **macOS**: `brew install ffmpeg`

### Высокое использование памяти

Модель требует ~2-3GB RAM. Для уменьшения нагрузки:
- Ограничьте количество воркеров (Gunicorn `-w 2`)
- Используйте swap память
- Увеличьте RAM сервера

### Медленная обработка

- Первый запрос всегда медленнее (загрузка модели)
- Используйте GPU для ускорения (требуется CUDA)
- Оптимизируйте размер аудиофайлов

## Структура файлов

```
scripts/
├── README.md                    # Этот файл
├── requirements.txt             # Python зависимости
├── download_model.py           # Скрипт загрузки модели
├── diarization_service.py      # Flask сервис
├── diarize.py                  # Локальный скрипт (fallback)
└── models/                     # Локальные модели
    └── pyannote/
        └── speaker-diarization-3.1/
```

## Лицензия

Модель pyannote.audio распространяется под лицензией MIT. Убедитесь, что вы приняли условия использования на HuggingFace.

## Ссылки

- [pyannote.audio](https://github.com/pyannote/pyannote-audio)
- [HuggingFace Model](https://huggingface.co/pyannote/speaker-diarization-3.1)
- [Flask Documentation](https://flask.palletsprojects.com/)
