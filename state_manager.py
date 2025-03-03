import streamlit as st

def initialize_session_state():
    """Initialize Streamlit session state variables."""
    if 'last_image_hash' not in st.session_state:
        st.session_state.last_image_hash = None
    if 'refresh_counter' not in st.session_state:
        st.session_state.refresh_counter = 0
    if 'current_image' not in st.session_state:
        st.session_state.current_image = None
    if 'current_caption' not in st.session_state:
        st.session_state.current_caption = None
    if 'extracted_images' not in st.session_state:
        st.session_state.extracted_images = []