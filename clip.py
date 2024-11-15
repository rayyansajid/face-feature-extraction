# from transformers import CLIPProcessor, CLIPModel, GPT2Tokenizer, GPT2LMHeadModel
# from PIL import Image
# import torch

# # Check for GPU availability
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # Load CLIP model and processor, move to GPU
# clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
# processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# # Load GPT-2 for text generation, move to GPU
# tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
# gpt2_model = GPT2LMHeadModel.from_pretrained("gpt2").to(device)

# # Function to generate description
# def generate_face_description(image_path):
#     # Open image and preprocess
#     image = Image.open(image_path)
#     inputs = processor(text=["a face"], images=image, return_tensors="pt", padding=True).to(device)
    
#     # Get CLIP embeddings
#     with torch.no_grad():
#         outputs = clip_model(**inputs)
#         image_features = outputs.image_embeds

#     # Generate face-focused prompt using GPT-2
#     prompt = "Describe this face in detail:"
#     input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
#     gpt_output = gpt2_model.generate(input_ids, max_length=50, num_return_sequences=1, pad_token_id=tokenizer.eos_token_id)
#     description = tokenizer.decode(gpt_output[0], skip_special_tokens=True)

#     return description

# # Example usage
# image_path = r"C:\Users\Rayyan Sajid\Downloads\face.jpeg"  # Update this path to your image
# description = generate_face_description(image_path)
# print("Generated Description:", description)

from transformers import CLIPProcessor, CLIPModel, GPT2Tokenizer, GPT2LMHeadModel
from PIL import Image
import torch

# Check for GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load CLIP model and processor, move to GPU
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Load GPT-2 for text generation, move to GPU
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
gpt2_model = GPT2LMHeadModel.from_pretrained("gpt2").to(device)

# Function to generate face-focused description
def generate_face_description(image_path):
    # Open image and preprocess
    image = Image.open(image_path)
    inputs = processor(text=["a person’s face with details"], images=image, return_tensors="pt", padding=True).to(device)
    
    # Get CLIP embeddings
    with torch.no_grad():
        outputs = clip_model(**inputs)
    
    # Detailed prompt for facial features
    prompt = ("Describe the facial features of the person in the image. "
              "Include details such as skin tone, eye shape, hair color, facial structure, "
              "and any unique characteristics or expressions.")
    prompt_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

    # Generate text using GPT-2
    gpt_output = gpt2_model.generate(
        prompt_ids, 
        max_length=150,
        num_return_sequences=1,
        temperature=0.5,  # Lower temperature for more focused response
        top_k=50,
        top_p=0.85,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )
    description = tokenizer.decode(gpt_output[0], skip_special_tokens=True)
    return description

# Example usage
image_path = r"C:\Users\Rayyan Sajid\Downloads\face.jpeg"  # Update this path to your image
description = generate_face_description(image_path)
print("Generated Description:", description)
