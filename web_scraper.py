import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import streamlit as st
from PIL import Image
from io import BytesIO

class WebScraper:
    def __init__(self):
        """Initialize web scraper with common headers."""
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        # Minimum dimensions for relevant images (in pixels)
        self.MIN_WIDTH = 200
        self.MIN_HEIGHT = 200

    def extract_images(self, url):
        """Extract images from website that meet size requirements."""
        try:
            if not self._is_valid_url(url):
                return [], "Invalid URL format"

            response = requests.get(url, headers=self.headers, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            images = []
            
            # Find all img tags
            for img in soup.find_all('img'):
                image_data = self._process_image(img, url)
                if image_data:
                    images.append(image_data)
            
            # Remove duplicates while preserving order
            unique_images = []
            seen_urls = set()
            for img in images:
                if img['url'] not in seen_urls:
                    seen_urls.add(img['url'])
                    unique_images.append(img)
            
            return unique_images, None
            
        except requests.exceptions.RequestException as e:
            return [], f"Error fetching website: {str(e)}"
        except Exception as e:
            return [], f"Error processing website: {str(e)}"

    def _process_image(self, img, base_url):
        """Process and validate individual image elements based on size."""
        # Get image source
        srcset = img.get('srcset', '').split()
        src = (img.get('src') or img.get('data-src') or img.get('data-original') or (srcset[0] if srcset else None))

        
        if not src:
            return None
            
        url = urljoin(base_url, src)
        if not self._is_valid_image_url(url):
            return None

        try:
            # Try to get image dimensions from HTML attributes first
            width = img.get('width')
            height = img.get('height')
            
            # Handle percentage-based dimensions
            if width and isinstance(width, str) and '%' in width:
                width = None
            if height and isinstance(height, str) and '%' in height:
                height = None
                
            # Convert string dimensions to integers if possible
            try:
                width = int(width) if width else None
                height = int(height) if height else None
            except (ValueError, TypeError):
                width = None
                height = None
            
            # If dimensions are in HTML and image is too small, skip it
            if width and height:
                if width < self.MIN_WIDTH or height < self.MIN_HEIGHT:
                    return None
            
            # If no valid dimensions in HTML, download and check image
            if not width or not height:
                # Download image
                response = requests.get(url, headers=self.headers, timeout=5)
                if response.status_code != 200:
                    return None
                
                # Open image and get its size
                img_content = BytesIO(response.content)
                with Image.open(img_content) as img_obj:
                    width, height = img_obj.size
                    
                    # Skip if image is too small
                    if width < self.MIN_WIDTH or height < self.MIN_HEIGHT:
                        return None

            # Get alt text and title
            alt_text = img.get('alt', '').strip()
            title = img.get('title', '').strip()
            
            return {
                'url': url,
                'alt': alt_text,
                'title': title,
                'width': width,
                'height': height
            }
            
        except Exception as e:
            st.error(f"Error processing image {url}: {str(e)}")
            return None

    def _is_valid_url(self, url):
        """Check if URL is valid."""
        try:
            result = urlparse(url)
            return all([result.scheme, result.netloc])
        except:
            return False

    def _is_valid_image_url(self, url):
        """Check if URL points to a valid image file."""
        image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.webp']
        url_lower = url.lower()
        return any(url_lower.endswith(ext) for ext in image_extensions)

    def download_image(self, image_url):
        """Download image from URL and return as bytes."""
        try:
            response = requests.get(image_url, headers=self.headers, timeout=10)
            response.raise_for_status()
            return response.content
        except Exception as e:
            st.error(f"Error downloading image from {image_url}: {str(e)}")
            return None
