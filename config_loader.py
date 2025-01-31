# config_loader.py
import logging
import torch

# Logger yapılandırması
logger = logging.getLogger(__name__)

def load_keys_from_file(filepath):
    keys = {}
    try:
        with open(filepath, 'r', encoding='utf-8') as file:
            lines = file.readlines()
            for line in lines:
                if '=' in line and not line.strip().startswith('#'):
                    key, value = line.strip().split('=', 1)
                    keys[key.strip()] = value.strip()
        logger.info(f"Config dosyası '{filepath}' başarıyla yüklendi.")
    except Exception as e:
        logger.error(f"Dosyadan anahtar okunurken hata oluştu: {e}")
    return keys

# Logger'ı yapılandır
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global cihaz seçimi
if torch.cuda.is_available():
    try:
        device = torch.device("cuda")
        torch.zeros(1).to(device)
        logger.info(f"Global device set to use: {device}")
    except Exception as e:
        logger.warning(f"CUDA detected but not usable. Falling back to CPU: {e}")
        device = torch.device("cpu")
else:
    device = torch.device("cpu")
    logger.info(f"Global device set to use: {device}")
