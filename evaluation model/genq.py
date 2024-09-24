from gtts import gTTS
import os

def text_to_speech(text, language='en', output_file='output.mp3'):
    """
    Convert text to speech and save as an MP3 file.
    
    :param text: The text to convert to speech
    :param language: The language of the text (default is English)
    :param output_file: The name of the output MP3 file
    """
    # Create a gTTS object
    tts = gTTS(text=text, lang=language, slow=False)
    
    # Save the audio file
    tts.save(output_file)
    
    print(f"Audio saved as {output_file}")
    
    # Play the audio file (works on macOS and Linux)
    # On Windows, you might need to use a different command or library
    os.system(f"play {output_file}")  # Requires SoX to be installed

# Example usage
if __name__ == "__main__":
    text = input("Enter the text you want to convert to speech: ")
    text_to_speech(text)