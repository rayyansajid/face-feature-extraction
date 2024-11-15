from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import torch

# Check for CUDA
print("CUDA available:", torch.cuda.is_available())
print("CUDA device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU detected")

# Load BLIP processor and model
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large").to("cuda")

# Load a local image
image_path = r"C:\Users\Rayyan Sajid\Downloads\face.jpeg"
raw_image = Image.open(image_path).convert('RGB')

# Conditional image captioning with enhanced prompt for more detailed output
# text = (
#     "Describe this person's face in detail, including characteristics like skin tone, hair color, eye shape, "
#     "nose structure, mouth shape, expression, age range, and any distinctive features such as freckles, scars, "
#     "or facial expressions."
# )
text = (
    "Describe only the person's facial characteristics in detail, such as face shape, eye color, eye size and shape, "
    "nose size and shape, mouth shape, lips, skin tone, hair color, hairstyle, and any facial expression. Avoid describing clothing or background."
)
inputs = processor(raw_image, text, return_tensors="pt").to("cuda")

# Generate caption with an increased max_length or max_new_tokens for detailed descriptions
# out = model.generate(**inputs, max_new_tokens=60)  # Adjust this number as needed
# out = model.generate(**inputs, max_new_tokens=80, num_beams=5, early_stopping=True)
out = model.generate(**inputs, max_new_tokens=80, temperature=0.7, top_k=50, num_beams=5, early_stopping=True)
print("Enhanced Conditional caption:", processor.decode(out[0], skip_special_tokens=True))

# Unconditional image captioning
inputs = processor(raw_image, return_tensors="pt").to("cuda")
out = model.generate(**inputs, max_new_tokens=60)  # Adjust this number as needed
print("Unconditional caption:", processor.decode(out[0], skip_special_tokens=True))
