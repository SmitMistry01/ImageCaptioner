# Install the transformers library
#!pip install transformers Pillow torch torchvision torchaudio

from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
# Initialize the processor and model from Hugging Face
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
# Load an image
image = Image.open("images.jpg")
# Prepare the image
inputs = processor(image, return_tensors="pt")
# Generate captions
outputs = model.generate(**inputs)
caption = processor.decode(outputs[0],skip_special_tokens=True)
 
print("Generated Caption:", caption)


# BlipProcessor: handles converting raw images into model-friendly tensors (pixel values, normalization, tokenization if text is involved).

# BlipForConditionalGeneration: the actual BLIP model that knows how to generate text (captions).

# PIL.Image: just for loading the image file.
# Image in → preprocessing → BLIP model → text tokens → human caption out.