# CUDA Setup Guide для Diarization Service

## Требования для GPU ускорения

### 1. Аппаратные требования
- NVIDIA GPU с поддержкой CUDA (GTX 10xx серии или новее)
- Минимум 4GB видеопамяти (рекомендуется 6GB+)
- Свободная видеопамять: минимум 2GB

### 2. Программные требования
- NVIDIA GPU драйверы (последняя версия)
- CUDA Toolkit 11.8 или 12.1 (опционально, PyTorch включает свою версию)
- Python 3.8+

## Пошаговая установка

### Шаг 1: Проверка GPU
```cmd
nvidia-smi
```
Убедитесь что команда работает и показывает вашу видеокарту.

### Шаг 2: Автоматическая установка (рекомендуется)
```cmd
setup_cuda.bat
```
Этот скрипт автоматически:
- Определит вашу CUDA версию
- Установит правильную версию PyTorch с CUDA
- Настроит все зависимости

### Шаг 3: Ручная установка PyTorch с CUDA
Если автоматическая установка не работает:

**Для CUDA 12.1:**
```cmd
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121
```

**Для CUDA 11.8:**
```cmd
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118
```

**Только CPU (без CUDA):**
```cmd
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
```

### Шаг 4: Проверка установки
```cmd
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0)}' if torch.cuda.is_available() else 'No GPU')"
```

## Проверка работы сервиса

### 1. Запуск сервиса
```cmd
start_service.bat
```

### 2. Проверка статуса GPU
```cmd
curl http://localhost:5000/system
```

### 3. Информация о здоровье
```cmd
curl http://localhost:5000/health
```

## Решение проблем

### GPU не обнаружен
1. Проверьте драйверы NVIDIA
2. Убедитесь что PyTorch установлен с CUDA поддержкой
3. Перезагрузите систему после установки драйверов

### Ошибка "CUDA out of memory"
1. Закройте другие программы использующие GPU
2. Уменьшите размер обрабатываемых файлов
3. Используйте меньший batch size (автоматически в pyannote)

### Медленная работа на GPU
1. Проверьте что модель действительно загружена на GPU
2. Убедитесь что нет конкурирующих процессов на GPU
3. Проверьте температуру GPU (nvidia-smi)

## Мониторинг производительности

### Проверка использования GPU во время работы
```cmd
nvidia-smi -l 1
```

### Детальная информация о памяти
```cmd
nvidia-smi --query-gpu=name,memory.total,memory.used,memory.free --format=csv
```

## Ожидаемая производительность

| Устройство | Время обработки (1 мин аудио) |
|-----------|-------------------------------|
| RTX 4090  | ~5-10 секунд                  |
| RTX 3080  | ~10-15 секунд                 |
| RTX 2070  | ~15-25 секунд                 |
| GTX 1070  | ~25-40 секунд                 |
| CPU i7    | ~60-120 секунд                |

*Время может варьироваться в зависимости от сложности аудио и количества говорящих*