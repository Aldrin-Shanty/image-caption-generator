import streamlit as st
from image_processor import ImageProcessor
from caption_generator import CaptionGenerator
from tts_engine import TextToSpeechEngine
from ui_manager import UIManager
from web_scraper import WebScraper
from state_manager import initialize_session_state

def main():
    st.set_page_config(
        page_title="Web-Enabled Image Caption Generator",
        layout="wide"
    )

    initialize_session_state()
    
    image_processor = ImageProcessor()
    caption_generator = CaptionGenerator()
    tts_engine = TextToSpeechEngine()
    web_scraper = WebScraper()
    ui_manager = UIManager(image_processor, caption_generator, tts_engine, web_scraper)
    
    ui_manager.render()

if __name__ == "__main__":
    main()

