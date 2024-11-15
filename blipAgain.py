# from transformers import BlipProcessor, BlipForConditionalGeneration
# from PIL import Image
# import torch

# # Check if CUDA is available and set the device
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # Load BLIP-2 model and processor, move to GPU
# processor = BlipProcessor.from_pretrained("Salesforce/blip2-flan-t5-xl")
# model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip2-flan-t5-xl").to(device)

# # Function to generate a face-focused description
# def generate_face_description(image_path):
#     # Open the image and process it
#     image = Image.open(image_path).convert("RGB")
#     inputs = processor(images=image, text="Describe the face in detail:", return_tensors="pt").to(device)

#     # Generate the description
#     with torch.no_grad():
#         output = model.generate(**inputs, max_length=100, num_beams=5, temperature=0.7)
#     description = processor.decode(output[0], skip_special_tokens=True)
    
#     return description

# # Example usage
image_path = r"C:\Users\Rayyan Sajid\Downloads\face.jpeg"  # Update this path to your image
# description = generate_face_description(image_path)
# print("Generated Description:", description)


# from transformers import Blip2ForConditionalGeneration, BlipProcessor
# from PIL import Image
# import torch

# # Load model and processor for BLIP-2
# model_name = "Salesforce/blip2-flan-t5-xl"
# model = Blip2ForConditionalGeneration.from_pretrained(model_name, torch_dtype=torch.float16).to("cuda" if torch.cuda.is_available() else "cpu")
# processor = BlipProcessor.from_pretrained(model_name)

# def generate_face_focused_description(image_path):
#     # Open and preprocess the image
#     image = Image.open(image_path).convert("RGB")
#     inputs = processor(images=image, return_tensors="pt").to("cuda" if torch.cuda.is_available() else "cpu")
    
#     # Generate description with a specific prompt for facial detail
    # prompt = "Describe the facial features of the person in the image. Include details such as skin tone, eye shape, hair color, facial structure, and any unique characteristics or expressions."
#     inputs.update(processor(text=prompt, return_tensors="pt", padding=True).to("cuda" if torch.cuda.is_available() else "cpu"))
#     outputs = model.generate(**inputs, max_length=50)
    
#     # Decode and return the generated text
#     description = processor.tokenizer.decode(outputs[0], skip_special_tokens=True)
#     return description

# # Path to your image
# # image_path = "path/to/your/image.jpg"  # Replace with the path to your image file

# # Generate the face-focused description
# description = generate_face_focused_description(image_path)
# print("Generated Face Description:", description)


##################################################
# from transformers import Blip2Processor, Blip2ForConditionalGeneration
# from PIL import Image
# import torch

# # Load model and processor
# model_name = "Salesforce/blip2-flan-t5-xl"
# device = "cuda" if torch.cuda.is_available() else "cpu"

# model = Blip2ForConditionalGeneration.from_pretrained(model_name).to(device)
# processor = Blip2Processor.from_pretrained(model_name)

# # Adjust processor for handling image token expansion
# processor.feature_extractor.expand_inputs_for_image_tokens = True
# processor.feature_extractor.return_special_image_tokens = True

# def generate_face_focused_description(image_path):
#     image = Image.open(image_path).convert("RGB")
#     prompt = "Describe the facial features of the person in the image. Include details such as skin tone, eye shape, hair color, facial structure, and any unique characteristics or expressions."
#     inputs = processor(images=image, text=prompt, return_tensors="pt").to(device)

#     # Generate the description with modified parameters
#     outputs = model.generate(
#         **inputs,
#         max_length=100,               # Increase for more detail
#         temperature=0.7,               # Slightly increase for more variety
#         top_p=0.9,                     # Use top-p sampling
#         num_return_sequences=1,        # Generate a single, more focused sequence
#     )
#     description = processor.tokenizer.decode(outputs[0], skip_special_tokens=True)
#     return description



# # Path to your image
# # image_path = "path/to/your/image.jpg"  # Replace with the path to your image file

# # Generate the face-focused description
# description = generate_face_focused_description(image_path)
# print("Generated Face Description:", description)



##################################################

import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image

# Load image
# image_path = "path/to/your/image.jpg"  # Replace with your actual image path
image = Image.open(image_path).convert("RGB")

# Load BLIP model and processor
model_name = "Salesforce/blip-image-captioning-base"
processor = BlipProcessor.from_pretrained(model_name)
model = BlipForConditionalGeneration.from_pretrained(model_name)

# Simplified prompt focusing on detailed facial description
prompt = "Describe the person’s facial features in detail."

# Process the image and generate input IDs
inputs = processor(image, prompt, return_tensors="pt")

# Set the generation parameters to encourage detailed responses
generation_kwargs = {
    "do_sample": True,
    "temperature": 0.5,
    "top_p": 0.7,
    "max_length": 100,  # Adjust max length for more detailed descriptions
}

# Generate the description
with torch.no_grad():
    try:
        output = model.generate(**inputs, **generation_kwargs)
        description = processor.decode(output[0], skip_special_tokens=True)
        print("Generated Face Description:", description)
    except Exception as e:
        print("An error occurred during generation:", e)

# CUDA check (if needed for debugging)
if torch.cuda.is_available():
    print("CUDA is available. Running on GPU.")
    print(f"CUDA Version: {torch.version.cuda}")
else:
    print("CUDA is not available. Running on CPU.")

