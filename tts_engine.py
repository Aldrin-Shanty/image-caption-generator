import streamlit as st
from gtts import gTTS
import base64
import tempfile


class TextToSpeechEngine:
    def speak(self, text):
        """Convert text to speech and auto-play it in Streamlit."""
        try:
            # Generate speech
            tts = gTTS(text=text, lang="en")

            # Save to a temporary file
            with tempfile.NamedTemporaryFile(delete=True, suffix=".mp3") as temp_audio:
                temp_filename = temp_audio.name
                tts.save(temp_filename)

                # Read file as bytes
                with open(temp_filename, "rb") as f:
                    audio_bytes = f.read()

            # Convert to Base64 for embedding
            audio_b64 = base64.b64encode(audio_bytes).decode()

            # Generate HTML + JavaScript to auto-play audio
            audio_html = f"""
            <audio id="tts-audio" autoplay>
                <source src="data:audio/mp3;base64,{audio_b64}" type="audio/mp3">
            </audio>
            <script>
                document.getElementById("tts-audio").play();
            </script>
            """
            st.markdown(audio_html, unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Error with text-to-speech: {str(e)}")


# Streamlit UI Example
if __name__ == "__main__":
    st.title("Text-to-Speech Demo")
    text_input = st.text_area("Enter text:")

    if st.button("Read Captions"):
        tts = TextToSpeechEngine()
        tts.speak(text_input)
