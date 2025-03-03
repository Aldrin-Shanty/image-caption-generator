import pyttsx3
import streamlit as st

class TextToSpeechEngine:
    def speak(self, text):
        """Convert text to speech."""
        try:
            engine = pyttsx3.init()
            self._configure_voice(engine)
            engine.say(text)
            engine.runAndWait()
            engine.stop()
        except Exception as e:
            st.error(f"Error with text-to-speech: {str(e)}")
        finally:
            try:
                engine.endLoop()
            except:
                pass

    def _configure_voice(self, engine):
        """Configure TTS engine settings."""
        voices = engine.getProperty('voices')
        selected_voice = self._select_preferred_voice(voices)
        if selected_voice:
            engine.setProperty('voice', selected_voice)
        engine.setProperty('rate', 145)
        engine.setProperty('volume', 1.0)

    def _select_preferred_voice(self, voices):
        """Select preferred voice from available options."""
        for voice in voices:
            if "david" in voice.name.lower():
                return voice.id
            elif "zira" in voice.name.lower():
                return voice.id
        return None
