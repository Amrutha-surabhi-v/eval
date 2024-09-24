import google.generativeai as genai
import speech_recognition as sr
import soundfile as sf
from api_k_rotate import get_next_api_key
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

def convert_to_wav(audio_file_path, output_file_path):
    data, samplerate = sf.read(audio_file_path)
    sf.write(output_file_path, data, samplerate, format='WAV')

def audio_to_text(audio_file):
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_file) as source:
        audio_data = recognizer.record(source)
        try:
            text = recognizer.recognize_google(audio_data)
            return text
        except sr.UnknownValueError:
            return "Could not understand the audio."
        except sr.RequestError:
            return "Error in the Google API request."

def get_key_answer(prompt):
    genai.configure(api_key=get_next_api_key())
    try:
        model = genai.GenerativeModel('gemini-pro')
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        print(f"Error occurred: {e}. Rotating to next API key.")
        return get_key_answer(prompt)  # Retry with the next key

def evaluate_student_answer(audio_file, question_prompt):
    student_answer = audio_to_text(audio_file)
    prompt = '''Evaluate the answer based on the question below. 
    Question: {}
    Answer: {}
    Please provide how accurate the student is and give if he is right or wrong.'''
    res = get_key_answer(prompt.format(question_prompt, student_answer))
    return res

if __name__ == "__main__":
    audio_file = 'output (1).wav' 
    question_prompt = "What is machine learning?"
    evaluation_result = evaluate_student_answer(audio_file, question_prompt)
    print("Evaluation Result:")
    print(evaluation_result)

