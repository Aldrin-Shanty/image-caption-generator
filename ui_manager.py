import streamlit as st
import pyperclip
import time

class UIManager:
    def __init__(self, image_processor, caption_generator, tts_engine, web_scraper):
        """Initialize UI manager with required components."""
        self.image_processor = image_processor
        self.caption_generator = caption_generator
        self.tts_engine = tts_engine
        self.web_scraper = web_scraper
        
    def render(self):
        """Render the main UI."""
        self.render_header()
        self.render_instructions()
        
        # Unified input section presenting all options together
        st.markdown('<h2 id="input-options-heading" tabindex="0">Input Options</h2>', unsafe_allow_html=True)
        st.markdown("""
            <p class="accessibility-note" aria-live="polite" tabindex="0">
                Choose any of the following methods to input images. Use Tab key to navigate between the options.
            </p>
        """, unsafe_allow_html=True)
        
        # 1. File Upload Option
        st.markdown("""
            <h3 id="upload-heading" tabindex="0">Option 1: Upload an Image File</h3>
        """, unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader(
            "Select an image file",
            type=['png', 'jpg', 'jpeg', 'gif', 'bmp'],
            help="Select an image file from your device. Supports PNG, JPG, JPEG, GIF, and BMP formats.",
            key="file_uploader"
        )
        
        if uploaded_file is not None:
            st.markdown(f"""
                <p class="sr-update" aria-live="assertive">
                    Image {getattr(uploaded_file, 'name', 'uploaded')} has been successfully uploaded and is being processed.
                </p>
            """, unsafe_allow_html=True)
            self.process_uploaded_file(uploaded_file)
        
        # 2. Clipboard Option (Windows only)
        st.markdown("""
            <h3 id="clipboard-heading" tabindex="0">Option 2: Get Image from Clipboard (Windows only)</h3>
        """, unsafe_allow_html=True)
        
        st.markdown("""
            <p class="clipboard-instruction" tabindex="0">
                If you have copied an image to your clipboard, you can use this option to process it.
                This feature only works on Windows.
            </p>
        """, unsafe_allow_html=True)
        
        if st.button(
            "Check Clipboard for Images", 
            key="clipboard_button",
            help="Click to check if any images are in your clipboard and process them."
        ):
            self.process_clipboard_image()
        
        # 3. Website URL Option
        st.markdown("""
            <h3 id="website-heading" tabindex="0">Option 3: Extract Images from Website</h3>
        """, unsafe_allow_html=True)
        
        st.markdown("""
            <p class="website-instruction" tabindex="0">
                Enter a website URL to extract and process all images found on that page.
            </p>
        """, unsafe_allow_html=True)
        
        url = st.text_input(
            "Enter website URL", 
            key="website_url",
            help="Type or paste a website URL here to extract images from that page."
        )
        
        if url and st.button(
            "Extract Images", 
            key="extract_button",
            help="Click to start extracting images from the provided website URL."
        ):
            st.markdown('<p aria-live="assertive">Beginning image extraction. This may take a moment...</p>', unsafe_allow_html=True)
            self.handle_website_extraction(url)
        
        # Always render image and caption at the end if there's any to display
        if st.session_state.current_image:
            self.render_image_and_caption()
        
        # Display all extracted images from website if available
        if 'extracted_images' in st.session_state and st.session_state.extracted_images:
            self.render_extracted_images()
        
        self.handle_refresh()

    def render_header(self):
        """Render application header."""
        st.markdown("""
            <h1 style='text-align: center;' tabindex="0">Accessible Image Caption Generator</h1>
        """, unsafe_allow_html=True)

    def render_instructions(self):
        """Render usage instructions with improved accessibility."""
        st.markdown("""
            <div role="region" aria-label="Instructions" tabindex="0">
                <h2 id="instructions-heading">Instructions:</h2>
                <ol>
                    <li>Choose any of the input methods below to provide an image</li>
                    <li>The application will automatically process your image and generate a caption</li>
                    <li>You can copy the caption or have it read aloud</li>
                    <li>Press Tab to navigate through elements and Space/Enter to activate buttons</li>
                    <li>Press Ctrl + R to refresh the page if needed</li>
                </ol>
            </div>
        """, unsafe_allow_html=True)

    def process_uploaded_file(self, uploaded_file):
        """Process uploaded image file."""
        img_bytes = uploaded_file.read()
        self.update_image_if_new(img_bytes, uploaded_file)

    def process_clipboard_image(self):
        """Process image from clipboard."""
        img_bytes = self.image_processor.get_clipboard_image()
        if img_bytes:
            st.markdown("""
                <p aria-live="assertive">
                    Image detected in clipboard and is now being processed.
                </p>
            """, unsafe_allow_html=True)
            self.update_image_if_new(img_bytes, img_bytes)
        else:
            st.warning("""No image detected in clipboard. Please copy an image to your clipboard first (Windows only).""")

    def update_image_if_new(self, img_bytes, image_source):
        """Update image and caption if new image detected."""
        current_hash = hash(img_bytes)
        if current_hash != st.session_state.last_image_hash:
            st.session_state.last_image_hash = current_hash
            st.session_state.current_image = img_bytes
            st.session_state.current_caption = self.caption_generator.generate_caption(image_source)
            st.markdown(f"""
                <p aria-live="assertive">
                    New image has been processed. Caption has been generated.
                </p>
            """, unsafe_allow_html=True)

    def render_image_and_caption(self):
        """Render image and caption with controls."""
        self.display_image_and_caption()

    def display_image_and_caption(self):
        """Display image, caption, and control buttons with improved accessibility."""
        st.markdown('<h2 id="results-heading" tabindex="0">Results</h2>', unsafe_allow_html=True)
        
        st.image(st.session_state.current_image, 
                caption=st.session_state.current_caption, 
                use_container_width=True)
        
        st.markdown('<h3 id="caption-heading" tabindex="0">Generated Caption:</h3>', unsafe_allow_html=True)
        with st.container():
            st.markdown(
                f'<div role="region" aria-labelledby="caption-heading" style="padding: 10px; '
                f'background-color: #e0e0e0; color: black; font-weight: bold; border-radius: 5px;" '
                f'tabindex="0">{st.session_state.current_caption}</div>',
                unsafe_allow_html=True
            )
            self.render_control_buttons()

    def render_control_buttons(self):
        """Render copy and TTS control buttons with improved accessibility."""
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Copy Caption to Clipboard", 
                        key="copy_button", 
                        help="Copies the generated caption to the clipboard. Use Tab to navigate to this button and Space to activate."):
                pyperclip.copy(st.session_state.current_caption)
                st.success("Caption copied to clipboard!")
                st.markdown('<p aria-live="assertive">Caption has been copied to clipboard successfully!</p>', unsafe_allow_html=True)
        with col2:
            if st.button("Read Caption Aloud", 
                        key="tts_button", 
                        help="Reads the generated caption aloud. Use Tab to navigate to this button and Space to activate."):
                self.tts_engine.speak(st.session_state.current_caption)
                st.markdown('<p aria-live="assertive">Caption is being read aloud now.</p>', unsafe_allow_html=True)

    def handle_website_extraction(self, url):
        """Handle website image extraction and processing."""
        images, error = self.web_scraper.extract_images(url)
        
        if error:
            st.error(error)
            st.markdown(f'<p aria-live="assertive">Error: {error}</p>', unsafe_allow_html=True)
        elif not images:
            st.warning("No images found on the website.")
            st.markdown('<p aria-live="assertive">No images were found on the provided website.</p>', unsafe_allow_html=True)
            st.session_state.extracted_images = []
        else:
            st.success(f"Found {len(images)} images!")
            st.markdown(f'<p aria-live="assertive">Successfully found {len(images)} images on the website.</p>', unsafe_allow_html=True)
            
            # Store extracted images in session state
            extracted_images_with_captions = []
            for idx, img_data in enumerate(images):
                # Download and process image
                img_bytes = self.web_scraper.download_image(img_data['url'])
                if img_bytes:
                    # Generate caption
                    caption = self.caption_generator.generate_caption(img_bytes)
                    extracted_images_with_captions.append({
                        'bytes': img_bytes,
                        'caption': caption,
                        'alt': img_data['alt'] or '',
                        'index': idx
                    })
            
            # Store in session state
            st.session_state.extracted_images = extracted_images_with_captions

    def render_extracted_images(self):
        """Render all extracted images with captions and controls."""
        if not st.session_state.extracted_images:
            return
        
        st.markdown('<h2 id="extracted-images-heading" tabindex="0">Extracted Images from Website</h2>', unsafe_allow_html=True)
        
        # Create expander for each image with improved accessibility
        for img_data in st.session_state.extracted_images:
            idx = img_data['index']
            expander_label = f"Image {idx + 1} {img_data['alt']}"
            with st.expander(expander_label):
                st.markdown(f'<div role="region" aria-label="{expander_label}" tabindex="0">', unsafe_allow_html=True)
                
                # Display image with alt text
                st.image(img_data['bytes'], caption=img_data['alt'], use_container_width=True)
                
                # Display caption with styling and accessibility attributes
                st.markdown('<h4 id="generated-caption-heading" tabindex="0">Generated Caption:</h4>', unsafe_allow_html=True)
                st.markdown(
                    f'<div role="region" aria-labelledby="generated-caption-heading" style="padding: 10px; '
                    f'background-color: #e0e0e0; color: black; font-weight: bold; border-radius: 5px;" '
                    f'tabindex="0">{img_data["caption"]}</div>',
                    unsafe_allow_html=True
                )
                
                # Add controls for each image with improved accessibility
                col1, col2 = st.columns(2)
                with col1:
                    if st.button(
                        "Copy Caption", 
                        key=f"copy_{idx}",
                        help="Copies this image's caption to clipboard. Use Tab to navigate and Space to activate."
                    ):
                        pyperclip.copy(img_data['caption'])
                        st.success("Caption copied!")
                        st.markdown('<p aria-live="assertive">Caption has been copied to clipboard successfully!</p>', unsafe_allow_html=True)
                with col2:
                    if st.button(
                        "Read Caption", 
                        key=f"read_{idx}",
                        help="Reads this image's caption aloud. Use Tab to navigate and Space to activate."
                    ):
                        self.tts_engine.speak(img_data['caption'])
                        st.markdown('<p aria-live="assertive">Caption is being read aloud now.</p>', unsafe_allow_html=True)
                
                st.markdown('</div>', unsafe_allow_html=True)

    def handle_refresh(self):
        """Handle periodic page refresh."""
        time.sleep(0.1)
        st.session_state.refresh_counter += 1
        if st.session_state.refresh_counter % 10 == 0:
            st.rerun()