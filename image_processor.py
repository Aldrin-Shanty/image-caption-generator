from PIL import Image
from io import BytesIO
import platform
import os
import streamlit as st

class ImageProcessor:
    def get_clipboard_image(self):
        """Get image from clipboard (Windows only)."""
        system = platform.system()
        if system == "Windows":
            from PIL import ImageGrab
            try:
                clipboard_data = ImageGrab.grabclipboard()
                if isinstance(clipboard_data, Image.Image):
                    return self._convert_image_to_bytes(clipboard_data)
                elif isinstance(clipboard_data, list) and len(clipboard_data) > 0:
                    return self._process_clipboard_file_list(clipboard_data)
            except Exception as e:
                st.error(f"Error accessing clipboard: {str(e)}")
                return None
        else:
            st.warning("Clipboard image detection is only supported on Windows.")
        return None

    def _convert_image_to_bytes(self, image):
        """Convert PIL Image to bytes."""
        img_byte_arr = BytesIO()
        image.save(img_byte_arr, format='PNG')
        return img_byte_arr.getvalue()

    def _process_clipboard_file_list(self, file_list):
        """Process list of files from clipboard."""
        for file_path in file_list:
            if self._is_valid_image_file(file_path):
                with Image.open(file_path) as img:
                    return self._convert_image_to_bytes(img)
        return None

    def _is_valid_image_file(self, file_path):
        """Check if file is a valid image file."""
        return (os.path.isfile(file_path) and 
                file_path.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')))
