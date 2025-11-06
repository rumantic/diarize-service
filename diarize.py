#!/usr/bin/env python3
"""
Скрипт для диаризации аудиофайла с использованием pyannote.audio
Возвращает JSON с временными метками говорящих
"""

import sys
import json
from pyannote.audio import Pipeline

def main():
    if len(sys.argv) < 2:
        print(json.dumps({"error": "No audio file provided"}))
        sys.exit(1)
    
    audio_file = sys.argv[1]
    
    try:
        # Инициализация pipeline для диаризации
        # Получите токен на https://huggingface.co/settings/tokens
        # и установите переменную окружения HF_TOKEN
        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=None  # Будет использован HF_TOKEN из окружения
        )
        
        # Выполнение диаризации
        diarization = pipeline(audio_file)
        
        # Формирование результата
        result = []
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            result.append({
                "start": float(turn.start),
                "end": float(turn.end),
                "speaker": speaker
            })
        
        print(json.dumps(result))
    
    except Exception as e:
        print(json.dumps({"error": str(e)}))
        sys.exit(1)

if __name__ == "__main__":
    main()
