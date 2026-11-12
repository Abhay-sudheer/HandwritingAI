from __future__ import annotations

import os
from pathlib import Path

import httpx
from PIL import Image


class Recognizer:
    def __init__(self):
        self.provider = os.getenv("INFERENCE_PROVIDER", "local").lower()
        self.model_path = Path(os.getenv("MODEL_PATH", "../models/trocr-base-handwritten"))
        if not self.model_path.is_absolute(): self.model_path = (Path(__file__).resolve().parents[3] / self.model_path).resolve()
        self.model = self.processor = self.device = None
        self.error = None
        self._load()

    def _load(self):
        if self.provider == "local":
            required = ["config.json", "preprocessor_config.json", "tokenizer_config.json", "vocab.json", "merges.txt"]
            if not self.model_path.is_dir() or any(not (self.model_path / name).is_file() for name in required) or not any((self.model_path / n).is_file() for n in ("model.safetensors", "pytorch_model.bin")):
                self.error = f"Local model files are incomplete at {self.model_path}."
                return
            try:
                import torch
                from transformers import TrOCRProcessor
                from importlib.metadata import version as package_version
                if int(package_version("transformers").split(".")[0]) >= 5:
                    from transformers import AutoModelForMultimodalLM
                    model_class = AutoModelForMultimodalLM
                else:
                    from transformers import VisionEncoderDecoderModel
                    model_class = VisionEncoderDecoderModel
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
                self.processor = TrOCRProcessor.from_pretrained(str(self.model_path), local_files_only=True)
                self.model = model_class.from_pretrained(str(self.model_path), local_files_only=True).to(self.device).eval()
            except Exception as exc:
                if isinstance(exc, OSError) and getattr(exc, "winerror", None) == 4551:
                    self.error = "Windows Application Control blocked a required PyTorch DLL (WinError 4551). Ask an authorized administrator to review Code Integrity policy; security controls were not changed."
                else:
                    self.error = f"Local model could not load ({type(exc).__name__}). Run scripts/diagnose.py for details."
        elif self.provider == "huggingface":
            if not os.getenv("HF_TOKEN"):
                self.error = "Hugging Face provider selected but HF_TOKEN is not configured."
        else:
            self.error = f"Unsupported inference provider: {self.provider}."

    @property
    def ready(self):
        return bool(self.model and self.processor) if self.provider == "local" else self.provider == "huggingface" and bool(os.getenv("HF_TOKEN"))

    def recognize_line(self, image: Image.Image) -> str:
        if self.provider == "huggingface":
            token = os.getenv("HF_TOKEN")
            if not token: raise RuntimeError("Hugging Face API token is not configured.")
            import io
            response = httpx.post("https://router.huggingface.co/hf-inference/models/" + os.getenv("HF_API_MODEL", "microsoft/trocr-base-handwritten"), headers={"Authorization": f"Bearer {token}"}, content=_png(image), timeout=90)
            response.raise_for_status()
            result = response.json()
            if isinstance(result, list) and result and isinstance(result[0], dict): return str(result[0].get("generated_text", ""))
            raise RuntimeError("The configured Hugging Face endpoint returned an unsupported response.")
        if not self.ready: raise RuntimeError(self.error or "Recognition model is not ready.")
        import torch
        pixel_values = self.processor(images=image, return_tensors="pt").pixel_values.to(self.device)
        with torch.inference_mode(): generated = self.model.generate(pixel_values, max_new_tokens=128)
        return self.processor.batch_decode(generated, skip_special_tokens=True)[0].strip()


def _png(image):
    import io
    buffer = io.BytesIO(); image.save(buffer, format="PNG"); return buffer.getvalue()
