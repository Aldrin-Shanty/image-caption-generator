from PIL import Image
from io import BytesIO
import platform
import os
import subprocess
import streamlit as st

class ImageProcessor:
    def get_clipboard_image(self):
        """Get image from clipboard (Windows & Linux)."""
        system = platform.system()
        if system == "Windows":
            return self._get_clipboard_image_windows()
        elif system == "Linux":
            return self._get_clipboard_image_linux()
        else:
            st.warning("Clipboard image detection is only supported on Windows and Linux.")
            return None

    def _get_clipboard_image_windows(self):
        """Retrieve clipboard image on Windows."""
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

    def _get_clipboard_image_linux(self):
        """Retrieve clipboard image on Linux using xclip or xsel."""
        try:
            process = subprocess.Popen(
                ["xclip", "-selection", "clipboard", "-t", "image/png", "-o"],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE
            )
            img_data, err = process.communicate()
            if process.returncode == 0 and img_data:
                return img_data
            
            # Try xsel if xclip fails
            process = subprocess.Popen(
                ["xsel", "--clipboard", "--output"],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE
            )
            img_data, err = process.communicate()
            if process.returncode == 0 and img_data:
                return img_data
            
        except FileNotFoundError:
            st.error("xclip or xsel is required for clipboard image support on Linux. Install with 'sudo apt install xclip' or 'sudo apt install xsel'.")
        except Exception as e:
            st.error(f"Error accessing clipboard: {str(e)}")
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

# Example usage
if __name__ == "__main__":
    processor = ImageProcessor()
    img_bytes = processor.get_clipboard_image()
    if img_bytes:
        st.image(img_bytes, caption="Clipboard Image", use_column_width=True)
