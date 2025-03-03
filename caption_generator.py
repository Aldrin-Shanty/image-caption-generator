import streamlit as st
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
from io import BytesIO
import torch

class CaptionGenerator:
    def __init__(self):
        torch.classes.__path__ = [] 
        """Initialize the BLIP image captioning model."""
        with st.spinner('Loading image captioning model... This may take a few minutes on first run...'):
            try:
                self.processor = BlipProcessor.from_pretrained(
                    "Salesforce/blip-image-captioning-base",
                    local_files_only=False
                )
                self.model = BlipForConditionalGeneration.from_pretrained(
                    "Salesforce/blip-image-captioning-base",
                    local_files_only=False
                )
                st.success("Model loaded successfully!")
            except Exception as e:
                st.error(f"Error loading model: {e}")
                self.processor = None
                self.model = None

    def generate_caption(self, image):
        """Generate caption for the given image."""
        if not self.processor or not self.model:
            return "Model not loaded properly. Please restart the application."
        try:
            if isinstance(image, bytes):
                image = Image.open(BytesIO(image))
            else:
                image = Image.open(image)
            inputs = self.processor(image, return_tensors="pt")
            output = self.model.generate(**inputs, max_length=50)
            return self.processor.decode(output[0], skip_special_tokens=True)
        except Exception as e:
            st.error(f"Error generating caption: {str(e)}")
            return "Unable to generate caption. Please try again."
