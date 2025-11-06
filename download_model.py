#!/usr/bin/env python3
"""
Скрипт для скачивания модели pyannote speaker-diarization локально
"""

import os
import sys
from dotenv import load_dotenv

# Загружаем переменные окружения из .env
load_dotenv()

def setup_proxy():
    """Настройка прокси из переменных окружения"""
    http_proxy = os.getenv('HTTP_PROXY') or os.getenv('http_proxy')
    https_proxy = os.getenv('HTTPS_PROXY') or os.getenv('https_proxy')
    no_proxy = os.getenv('NO_PROXY') or os.getenv('no_proxy')
    
    proxies = {}
    
    if http_proxy or https_proxy:
        print("\n" + "="*70)
        print("Proxy Configuration:")
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
            print(f"  NO_PROXY: {no_proxy}")
            os.environ['no_proxy'] = no_proxy
            os.environ['NO_PROXY'] = no_proxy
        print("="*70 + "\n")
        
        # Настраиваем прокси для requests/urllib
        try:
            import urllib.request
            proxy_handler = urllib.request.ProxyHandler(proxies)
            opener = urllib.request.build_opener(proxy_handler)
            urllib.request.install_opener(opener)
            print("✓ Proxy configured for urllib")
        except Exception as e:
            print(f"⚠ Warning: Could not configure urllib proxy: {e}")
    
    return proxies
    
def main():
    # Настройка прокси ПЕРЕД импортом pyannote
    proxies = setup_proxy()
    
    # Путь для сохранения модели
    model_path = os.getenv('PYANNOTE_MODEL_PATH', './models/pyannote-speaker-diarization-3.1')
    hf_token = os.getenv('HF_TOKEN')

    if not hf_token:
        print("ERROR: HF_TOKEN environment variable not set!")
        print("Get your token from https://huggingface.co/settings/tokens")
        sys.exit(1)

    print(f"Downloading model to: {model_path}")
    print("\n" + "="*70)
    print("IMPORTANT: Make sure you accepted licenses for these models:")
    print("  1. https://huggingface.co/pyannote/speaker-diarization-3.1")
    print("  2. https://huggingface.co/pyannote/segmentation-3.0")
    print("  3. https://huggingface.co/pyannote/wespeaker-voxceleb-resnet34-LM")
    print("="*70 + "\n")

    try:
        # Настраиваем прокси для huggingface_hub
        if proxies:
            print("Configuring HuggingFace Hub to use proxy...")
            from huggingface_hub import configure_http_backend
            import requests
            
            # Создаем сессию requests с прокси
            session = requests.Session()
            session.proxies.update(proxies)
            session.verify = True  # Проверяем SSL сертификаты
            
            # Настраиваем backend для huggingface_hub
            try:
                configure_http_backend(backend_factory=lambda: session)
                print("✓ HuggingFace Hub configured with proxy")
            except Exception as e:
                print(f"⚠ Warning: Could not configure HF Hub backend: {e}")
                print("Continuing with environment proxy settings...")
        
        from pyannote.audio import Pipeline
        
        # Создаём директорию если не существует
        os.makedirs(model_path, exist_ok=True)

        # Загружаем модель с HuggingFace
        print("Loading model from HuggingFace...")
        print(f"Using HF token: {hf_token[:10]}...")
        
        # Устанавливаем токен через стандартную переменную окружения HuggingFace
        os.environ['HF_TOKEN'] = hf_token
        
        # Увеличиваем таймауты для медленного интернета/прокси
        os.environ['HF_HUB_DOWNLOAD_TIMEOUT'] = '600'
        os.environ['HF_HUB_ENABLE_HF_TRANSFER'] = '1'
        
        # Загружаем без явной передачи токена - будет использован из env
        print("Downloading... (this may take several minutes)")
        pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1")

        # Сохраняем локально
        print("Saving model locally...")
        pipeline.save_pretrained(model_path)

        print(f"\n✓ Model successfully downloaded to {model_path}")
        print("You can now run the diarization service without internet connection!")

    except Exception as e:
        print(f"\n✗ Error downloading model: {str(e)}")
        print("\nTroubleshooting:")
        print("1. Accept all model licenses:")
        print("   - https://huggingface.co/pyannote/speaker-diarization-3.1")
        print("   - https://huggingface.co/pyannote/segmentation-3.0")
        print("   - https://huggingface.co/pyannote/wespeaker-voxceleb-resnet34-LM")
        print("2. Check your HF token is valid: https://huggingface.co/settings/tokens")
        print("3. Make sure you have internet connection")
        print("4. If using proxy, check proxy settings in .env")
        sys.exit(1)

if __name__ == "__main__":
    main()
