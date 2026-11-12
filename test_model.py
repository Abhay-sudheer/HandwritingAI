
print("TEST STARTED")

from transformers import TrOCRProcessor, VisionEncoderDecoderModel

print("Transformers imported successfully!")

model_path = r"C:\Users\ABHAY\OneDrive\Desktop\Clg_Mini_Project\models\trocr-base-handwritten"

print("Loading processor...")
processor = TrOCRProcessor.from_pretrained(model_path)

print("Loading model...")
model = VisionEncoderDecoderModel.from_pretrained(model_path)

print("\nSUCCESS! TrOCR model loaded!")
print("Parameters:", sum(p.numel() for p in model.parameters()))
