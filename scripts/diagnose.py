from __future__ import annotations
import importlib.metadata as metadata
import platform
import sys
from pathlib import Path
import os
ROOT = Path(__file__).resolve().parents[1]
MODEL = Path(os.getenv('MODEL_PATH', str(ROOT.parent / 'models' / 'trocr-base-handwritten')))
if not MODEL.is_absolute(): MODEL = (ROOT / MODEL).resolve()
print('Python:', platform.python_version(), sys.executable)
print('Virtual environment:', sys.prefix)
for name in ('torch','torchvision','transformers','fastapi','uvicorn','opencv-python-headless','Pillow','numpy'):
    try: print(f'{name}: {metadata.version(name)}')
    except metadata.PackageNotFoundError: print(f'{name}: MISSING')
print('Model directory:', MODEL, 'exists=', MODEL.is_dir())
for name in ('config.json','generation_config.json','model.safetensors','pytorch_model.bin','preprocessor_config.json','tokenizer_config.json','vocab.json','merges.txt'):
    print(f'  {name}: {(MODEL/name).is_file()}')
for name in ('torch','torchvision','transformers','fastapi','cv2','PIL','numpy'):
    try:
        module=__import__(name); print(f'{name} import: OK {getattr(module,"__version__","")}')
    except Exception as exc: print(f'{name} import: FAILED {type(exc).__name__}: {exc}')
try:
    from transformers import TrOCRProcessor
    from importlib.metadata import version as package_version
    if int(package_version('transformers').split('.')[0]) >= 5:
        from transformers import AutoModelForMultimodalLM as ModelClass
    else:
        from transformers import VisionEncoderDecoderModel as ModelClass
    print('TrOCR loading: attempting local files only')
    TrOCRProcessor.from_pretrained(str(MODEL), local_files_only=True)
    ModelClass.from_pretrained(str(MODEL), local_files_only=True)
    print('TrOCR loading: OK')
except Exception as exc: print(f'TrOCR loading: FAILED {type(exc).__name__}: {exc}')
