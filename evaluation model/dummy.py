# import pyaudio
# import wave
# import numpy as np
# import keyboard

# # Parameters
# FORMAT = pyaudio.paInt16  # Audio format
# CHANNELS = 1  # Number of audio channels (1 for mono, 2 for stereo)
# RATE = 44100  # Sampling rate (in Hz)
# CHUNK = 1024  # Number of frames per buffer
# RECORDING_FILENAME = "audio_rec.mp3"

# # Initialize PyAudio
# audio = pyaudio.PyAudio()

# # Create a stream
# stream = audio.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)

# print("Recording... Press 's' to stop recording.")

# frames = []

# try:
#     while True:
#         # Read audio data from the stream
#         data = stream.read(CHUNK)
        
#         # Convert data to numpy array for processing
#         numpy_data = np.frombuffer(data, dtype=np.int16)
        
#         # Simple noise reduction
#         # Apply a threshold to remove low amplitude signals (noise)
#         numpy_data = np.where(np.abs(numpy_data) < 500, 0, numpy_data)
        
#         # Convert numpy array back to bytes and store it
#         frames.append(numpy_data.tobytes())
        
#         # Check if 's' is pressed to stop recording
#         if keyboard.is_pressed('s'):
#             print("Stopping...")
#             break
# finally:
#     # Stop and close the stream
#     stream.stop_stream()
#     stream.close()
    
#     # Terminate PyAudio
#     audio.terminate()

#     # Save the recorded audio to a file
#     with wave.open(RECORDING_FILENAME, 'wb') as wf:
#         wf.setnchannels(CHANNELS)
#         wf.setsampwidth(audio.get_sample_size(FORMAT))
#         wf.setframerate(RATE)
#         wf.writeframes(b''.join(frames))

# print(f"Recording saved to {RECORDING_FILENAME}")




# from transformers import BertModel, BertTokenizer
# import torch
# from sklearn.metrics.pairwise import cosine_similarity

# def bert_similarity_evaluation(key_answer, student_answer):
#     # Load pre-trained BERT model and tokenizer
#     model = BertModel.from_pretrained('bert-base-uncased')
#     tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

#     # Tokenize the input text and get embeddings
#     def get_embedding(text):
#         inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True)
#         with torch.no_grad():
#             outputs = model(**inputs)
#         return outputs.last_hidden_state.mean(dim=1).squeeze()

#     # Get embeddings for both answers
#     key_embedding = get_embedding(key_answer)
#     student_embedding = get_embedding(student_answer)

#     # Compute cosine similarity
#     similarity_score = cosine_similarity(key_embedding.unsqueeze(0), student_embedding.unsqueeze(0))
    
#     return similarity_score[0][0]

# # Example Usage
# key_answer = "Artificial intelligence is the simulation of human intelligence by machines."
# student_answer = "AI involves machines simulating human-like intelligence."

# similarity_score = bert_similarity_evaluation(key_answer, student_answer)
# print(f"BERT Similarity Score: {similarity_score}")



'''AUDIORECORDING'''




# import pyaudio
# import wave
# import numpy as np
# import keyboard
# from scipy.signal import butter, lfilter

# # Parameters
# FORMAT = pyaudio.paInt16  # Audio format
# CHANNELS = 1  # Number of audio channels (1 for mono, 2 for stereo)ss
# RATE = 44100  # Sampling rate (in Hz)
# CHUNK = 1024  # Number of frames per buffer
# RECORDING_FILENAME = "audio_rec.wav"

# # Bandpass filter parameters
# LOWCUT = 300.0  # Lower bound of the frequency range (300 Hz)
# HIGHCUT = 3400.0  # Upper bound of the frequency range (3400 Hz)

# # Initialize PyAudio
# audio = pyaudio.PyAudio()

# # Create a stream
# stream = audio.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)

# print("Recording... Press 's' to stop recording.")

# frames = []

# def butter_bandpass(lowcut, highcut, fs, order=5):
#     nyquist = 0.5 * fs
#     low = lowcut / nyquist
#     high = highcut / nyquist
#     b, a = butter(order, [low, high], btype='band')
#     return b, a

# def apply_bandpass_filter(data, lowcut, highcut, fs, order=5):
#     b, a = butter_bandpass(lowcut, highcut, fs, order=order)
#     y = lfilter(b, a, data)
#     return y

# def smooth(data, window_size=5):
#     return np.convolve(data, np.ones(window_size)/window_size, mode='same')

# try:
#     while True:
#         # Read audio data from the stream
#         data = stream.read(CHUNK)
        
#         # Convert data to numpy array for processing
#         numpy_data = np.frombuffer(data, dtype=np.int16)
        
#         # Apply a bandpass filter (retaining frequencies in human voice range)
#         filtered_data = apply_bandpass_filter(numpy_data, LOWCUT, HIGHCUT, RATE)
        
#         # Apply a simple smoothing filter to reduce noise spikes
#         smoothed_data = smooth(filtered_data)
        
#         # Apply a noise gate (optional: more refined than a simple threshold)
#         noise_gate_threshold = 500
#         gated_data = np.where(np.abs(smoothed_data) < noise_gate_threshold, 0, smoothed_data)
        
#         # Convert numpy array back to bytes and store it
#         frames.append(gated_data.astype(np.int16).tobytes())
        
#         # Check if 's' is pressed to stop recording
#         if keyboard.is_pressed('s'):
#             print("Stopping...")
#             break
# finally:
#     # Stop and close the stream
#     stream.stop_stream()
#     stream.close()
    
#     # Terminate PyAudio
#     audio.terminate()

#     # Save the recorded audio to a file (WAV format)
#     with wave.open(RECORDING_FILENAME, 'wb') as wf:
#         wf.setnchannels(CHANNELS)
#         wf.setsampwidth(audio.get_sample_size(FORMAT))
#         wf.setframerate(RATE)
#         wf.writeframes(b''.join(frames))

# print(f"Recording saved to {RECORDING_FILENAME}")



# import openai
# import speech_recognition as sr
# from transformers import BertModel, BertTokenizer
# import torch
# from sklearn.metrics.pairwise import cosine_similarity
# from pydub import AudioSegment

# def convert_to_wav(audio_file_path, output_file_path):
#     audio = AudioSegment.from_file(audio_file_path)
#     audio.export(output_file_path, format='wav')

# def audio_to_text(audio_file):
#     recognizer = sr.Recognizer()
#     with sr.AudioFile(audio_file) as source:
#         audio_data = recognizer.record(source)
#         try:
#             text = recognizer.recognize_google(audio_data)
#             return text
#         except sr.UnknownValueError:
#             return "Could not understand the audio."
#         except sr.RequestError:
#             return "Error in the Google API request."

# def get_key_answer(prompt):
#     openai.api_key = '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'

#     response = openai.ChatCompletion.create(
#         model="gpt-3.5-turbo",  # Use an appropriate model
#         messages=[{"role": "user", "content": prompt}]
#     )
#     return response['choices'][0]['message']['content'].strip()


# # Function for BERT-based similarity evaluation
# def bert_similarity_evaluation(key_answer, student_answer):
#     # Load pre-trained BERT model and tokenizer
#     model = BertModel.from_pretrained('bert-base-uncased')
#     tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

#     # Tokenize the input text and get embeddings
#     def get_embedding(text):
#         inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True)
#         with torch.no_grad():
#             outputs = model(**inputs)
#         return outputs.last_hidden_state.mean(dim=1).squeeze()

#     # Get embeddings for both answers
#     key_embedding = get_embedding(key_answer)
#     student_embedding = get_embedding(student_answer)

#     # Compute cosine similarity
#     similarity_score = cosine_similarity(key_embedding.unsqueeze(0), student_embedding.unsqueeze(0))
    
#     return similarity_score[0][0]

# # Main function to handle audio input and evaluate similarity
# def evaluate_student_answer(audio_file, question_prompt):
#     # Convert audio to text
#     student_answer = audio_to_text(audio_file)
#     print(f"Transcribed Student Answer: {student_answer}")

#     # If the transcription was successful
#     if "Could not understand the audio" not in student_answer and "Error in the Google API request" not in student_answer:
#         # Get the key answer from OpenAI based on the question prompt
#         key_answer = get_key_answer(question_prompt)
#         print(f"Generated Key Answer: {key_answer}")

#         # Perform BERT similarity evaluation
#         similarity_score = bert_similarity_evaluation(key_answer, student_answer)
#         print(f"Similarity Score: {similarity_score}")

#         # Check if similarity is above 60%
#         if similarity_score >= 0.60:
#             print("The answer given by the student is correct.")
#         else:
#             print("The answer given by the student is incorrect.")
#     else:
#         print("Transcription failed, unable to evaluate the answer.")

# # Example Usage
# audio_file = 'evaluation model\\output.wav'  
# question_prompt = "What is machine learning?"  

# evaluate_student_answer(audio_file, question_prompt)





# import openai
# import speech_recognition as sr
# from transformers import BertModel, BertTokenizer
# import torch
# from sklearn.metrics.pairwise import cosine_similarity
# from pydub import AudioSegment

# def convert_to_wav(audio_file_path, output_file_path):
#     audio = AudioSegment.from_file(audio_file_path)
#     audio.export(output_file_path, format='wav')

# def audio_to_text(audio_file):
#     recognizer = sr.Recognizer()
#     with sr.AudioFile(audio_file) as source:
#         audio_data = recognizer.record(source)
#         try:
#             text = recognizer.recognize_google(audio_data)
#             return text
#         except sr.UnknownValueError:
#             return "Could not understand the audio."
#         except sr.RequestError:
#             return "Error in the Google API request."

# # # Function to get key answer from OpenAI GPT-3
# # def get_key_answer(prompt):
# #     openai.api_key = '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'  # Replace with your actual API key
# #     response = openai.ChatCompletion.create(
# #         model="gpt-3.5-turbo",  # Use the correct model
# #         messages=[{"role": "user", "content": prompt}]
# #     )
# #     return response['choices'][0]['message']['content'].strip()

# # # Function for BERT-based similarity evaluation
# def bert_similarity_evaluation(key_answer, student_answer):
#     # Load pre-trained BERT model and tokenizer
#     model = BertModel.from_pretrained('bert-base-uncased')
#     tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

#     # Tokenize the input text and get embeddings
#     def get_embedding(text):
#         inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True)
#         with torch.no_grad():
#             outputs = model(**inputs)
#         return outputs.last_hidden_state.mean(dim=1).squeeze()

#     # Get embeddings for both answers
#     key_embedding = get_embedding(key_answer)
#     student_embedding = get_embedding(student_answer)

#     # Compute cosine similarity
#     similarity_score = cosine_similarity(key_embedding.unsqueeze(0), student_embedding.unsqueeze(0))
    
#     return similarity_score[0][0]

# # Main function to handle audio input and evaluate similarity
# def evaluate_student_answer(audio_file, question_prompt):
#     # Convert audio to text
#     student_answer = audio_to_text(audio_file)
#     print(f"Transcribed Student Answer: {student_answer}")

#     # If the transcription was successful
#     if "Could not understand the audio" not in student_answer and "Error in the Google API request" not in student_answer:
#         # Get the key answer from OpenAI based on the question prompt
#         key_answer = get_key_answer(question_prompt)
#         print(f"Generated Key Answer: {key_answer}")

#         # Perform BERT similarity evaluation
#         similarity_score = bert_similarity_evaluation(key_answer, student_answer)
#         print(f"Similarity Score: {similarity_score}")

#         # Check if similarity is above 60%
#         if similarity_score >= 0.60:
#             print("The answer given by the student is correct.")
#         else:
#             print("The answer given by the student is incorrect.")
#     else:
#         print("Transcription failed, unable to evaluate the answer.")

# # Example Usage
# audio_file = 'evaluation model/output.wav'  
# question_prompt = "What is machine learning?"  

# evaluate_student_answer(audio_file, question_prompt)


# import random
# import openai
# import speech_recognition as sr
# from transformers import BertModel, BertTokenizer
# import torch
# from sklearn.metrics.pairwise import cosine_similarity
# from pydub import AudioSegment

# # API key rotation setup
# API_KEYS = [
#     "key1_abcdefg",
#     "key2_hijklmn",
#     "key3_opqrstu",
#     "key4_vwxyz12"
# ]

# def create_rotator(keys, rotation_interval=3):
#     return {
#         "keys": keys,
#         "rotation_interval": rotation_interval,
#         "current_index": 0,
#         "request_count": 0
#     }

# def rotate(rotator):
#     rotator["request_count"] += 1
#     if rotator["request_count"] % rotator["rotation_interval"] == 0:
#         rotator["current_index"] = (rotator["current_index"] + 1) % len(rotator["keys"])
#     return rotator["keys"][rotator["current_index"]]

# # Create a global rotator
# api_rotator = create_rotator(API_KEYS)

# # ... (keep your existing functions for convert_to_wav and audio_to_text)

# # Modified function to get key answer from OpenAI GPT-3 with rotating API keys
# def get_key_answer(prompt):
#     openai.api_key = rotate(api_rotator)
#     try:
#         response = openai.ChatCompletion.create(
#             model="gpt-3.5-turbo",
#             messages=[{"role": "user", "content": prompt}]
#         )
#         return response['choices'][0]['message']['content'].strip()
#     except openai.error.RateLimitError:
#         print("Rate limit exceeded. Rotating to next API key.")
#         return get_key_answer(prompt)  # Retry with the next key

# # ... (keep your existing functions for bert_similarity_evaluation)

# # Main function to handle audio input and evaluate similarity
# def evaluate_student_answer(audio_file, question_prompt):
#     # Convert audio to text
#     student_answer = audio_to_text(audio_file)
#     print(f"Transcribed Student Answer: {student_answer}")
    
#     # If the transcription was successful
#     if "Could not understand the audio" not in student_answer and "Error in the Google API request" not in student_answer:
#         # Get the key answer from OpenAI based on the question prompt
#         key_answer = get_key_answer(question_prompt)
#         print(f"Generated Key Answer: {key_answer}")
        
#         # Perform BERT similarity evaluation
#         similarity_score = bert_similarity_evaluation(key_answer, student_answer)
#         print(f"Similarity Score: {similarity_score}")
        
#         # Check if similarity is above 60%
#         if similarity_score >= 0.60:
#             print("The answer given by the student is correct.")
#         else:
#             print("The answer given by the student is incorrect.")
#     else:
#         print("Transcription failed, unable to evaluate the answer.")

# # Example Usage
# audio_file = 'evaluation model/output.wav' 
# question_prompt = "What is machine learning?"
# evaluate_student_answer(audio_file, question_prompt)



# #using circular queue
# import random
# from typing import List

# class APIKeyRotator:
#     def __init__(self, api_keys: List[str]):
#         self.api_keys = api_keys

#     def get_random_key(self) -> str:
#         return random.choice(self.api_keys)

# def create_api_key_rotator(api_keys: List[str]) -> callable:
#     rotator = APIKeyRotator(api_keys)
#     return rotator.get_random_key

# api_keys = ["key1", "key2", "key3", "key4", "key5"]
# get_random_api_key = create_api_key_rotator(api_keys)

# for _ in range(7):
#     print(get_random_api_key())



#using double ended queue

# from typing import List
# from collections import deque
# import random

# class APIKeyRotator:
#     def _init_(self, api_keys: List[str]):
#         self.api_keys = deque(api_keys)
#         self.current_key_index = 0
#     def get_random_key(self) -> str:
#         return random.choice(self.api_keys)

# def get_next_key(self) -> str:
#         current_key = self.api_keys[0]
#         self.api_keys.rotate(-1)
#         return current_key

# def create_api_key_rotator(api_keys: List[str]) -> callable:
#     rotator = APIKeyRotator(api_keys)
#     return rotator.get_next_key


# api_keys = ["key1", "key2", "key3", "key4", "key5"]
# get_random_api_key = create_api_key_rotator(api_keys)




# 
# import random

# API_KEYS = ["",""]
    


# def create_rotator(keys, rotation_interval=3):
#     return {
#         "keys": keys,
#         "rotation_interval": rotation_interval,
#         "current_index": 0,
#         "request_count": 0
#     }

# def rotate(rotator):
#     rotator["request_count"] += 1
#     if rotator["request_count"] % rotator["rotation_interval"] == 0:
#         rotator["current_index"] = (rotator["current_index"] + 1) % len(rotator["keys"])
#     return rotator["keys"][rotator["current_index"]]

# def get_current_key(rotator):
#     return rotator["keys"][rotator["current_index"]]

# def simulate_api_request(rotator):
#     key = rotate(rotator)
#     # Simulate some randomness in API response time
#     response_time = random.uniform(0.1, 0.5)
#     return f"Using API key: {key} (Response time: {response_time:.2f}s)"

# # Usage
# rotator = create_rotator(API_KEYS)

# for i in range(10):
#     result = simulate_api_request(rotator)
#     print(f"Request {i+1}: {result}")



'''API roatation with the evaluation in same code'''
# import random
# import openai
# import speech_recognition as sr
# from transformers import BertModel, BertTokenizer
# import torch
# from sklearn.metrics.pairwise import cosine_similarity
# from pydub import AudioSegment

# # API key rotation setup
# API_KEYS = [
#     "AIzaSyB0u9m1csGnS6HAT2O5iU5N0YK-jrAaBPQ",
#     "AIzaSyBofeEmwj09yMB2Vo6TwT2ibKAYfC2hxH0",
    
# ]

# def create_rotator(keys, rotation_interval=3):
#     return {
#         "keys": keys,
#         "rotation_interval": rotation_interval,
#         "current_index": 0,
#         "request_count": 0
#     }

# def rotate(rotator):
#     rotator["request_count"] += 1
#     if rotator["request_count"] % rotator["rotation_interval"] == 0:
#         rotator["current_index"] = (rotator["current_index"] + 1) % len(rotator["keys"])
#     return rotator["keys"][rotator["current_index"]]

# # Create a global rotator
# api_rotator = create_rotator(API_KEYS)
 
# def convert_to_wav(audio_file_path, output_file_path):
#     audio = AudioSegment.from_file(audio_file_path)
#     audio.export(output_file_path, format='wav')

# def audio_to_text(audio_file):
#     recognizer = sr.Recognizer()
#     with sr.AudioFile(audio_file) as source:
#         audio_data = recognizer.record(source)
#         try:
#             text = recognizer.recognize_google(audio_data)
#             return text
#         except sr.UnknownValueError:
#             return "Could not understand the audio."
#         except sr.RequestError:
#             return "Error in the Google API request."
# def get_key_answer(prompt):
#     openai.api_key = rotate(api_rotator)
#     try:
#         response = openai.ChatCompletion.create(
#             model="gpt-3.5-turbo",
#             messages=[{"role": "user", "content": prompt}]
#         )
#         return response['choices'][0]['message']['content'].strip()
#     except openai.error.RateLimitError:
#         print("Rate limit exceeded. Rotating to next API key.")
#         return get_key_answer(prompt)  # Retry with the next key

# def bert_similarity_evaluation(key_answer, student_answer):
#     # Load pre-trained BERT model and tokenizer
#     model = BertModel.from_pretrained('bert-base-uncased')
#     tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

#     # Tokenize the input text and get embeddings
#     def get_embedding(text):
#         inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True)
#         with torch.no_grad():
#             outputs = model(**inputs)
#         return outputs.last_hidden_state.mean(dim=1).squeeze()

#     # Get embeddings for both answers
#     key_embedding = get_embedding(key_answer)
#     student_embedding = get_embedding(student_answer)

#     # Compute cosine similarity
#     similarity_score = cosine_similarity(key_embedding.unsqueeze(0), student_embedding.unsqueeze(0))
    
#     return similarity_score[0][0]

# # Main function to handle audio input and evaluate similarity
# def evaluate_student_answer(audio_file, question_prompt):
#     # Convert audio to text
#     student_answer = audio_to_text(audio_file)
#     print(f"Transcribed Student Answer: {student_answer}")
    
#     # If the transcription was successful
#     if "Could not understand the audio" not in student_answer and "Error in the Google API request" not in student_answer:
#         # Get the key answer from OpenAI based on the question prompt
#         key_answer = get_key_answer(question_prompt)
#         print(f"Generated Key Answer: {key_answer}")
        
#         # Perform BERT similarity evaluation
#         similarity_score = bert_similarity_evaluation(key_answer, student_answer)
#         print(f"Similarity Score: {similarity_score}")
        
#         # Check if similarity is above 60%
#         if similarity_score >= 0.60:
#             print("The answer given by the student is correct.")
#         else:
#             print("The answer given by the student is incorrect.")
#     else:
#         print("Transcription failed, unable to evaluate the answer.")

# # Example Usage
# audio_file = 'evaluation model/output.wav' 
# question_prompt = "What is machine learning?"
# evaluate_student_answer(audio_file, question_prompt)
'''this is api seperaete and function seperate with openai key functions'''
# import openai
# import speech_recognition as sr
# from transformers import BertModel, BertTokenizer
# import torch
# from sklearn.metrics.pairwise import cosine_similarity
# from pydub import AudioSegment
# from api_k_rotate import get_next_api_key

# def convert_to_wav(audio_file_path, output_file_path):
#     audio = AudioSegment.from_file(audio_file_path)
#     audio.export(output_file_path, format='wav')

# def audio_to_text(audio_file):
#     recognizer = sr.Recognizer()
#     with sr.AudioFile(audio_file) as source:
#         audio_data = recognizer.record(source)
#         try:
#             text = recognizer.recognize_google(audio_data)
#             return text
#         except sr.UnknownValueError:
#             return "Could not understand the audio."
#         except sr.RequestError:
#             return "Error in the Google API request."

# def get_key_answer(prompt):
#     openai.api_key = get_next_api_key()
#     try:
#         response = openai.ChatCompletion.create(
#             model="gpt-3.5-turbo",
#             messages=[{"role": "user", "content": prompt}]
#         )
#         return response['choices'][0]['message']['content'].strip()
#     except openai.error.RateLimitError:
#         print("Rate limit exceeded. Rotating to next API key.")
#         return get_key_answer(prompt)  # Retry with the next key

# def bert_similarity_evaluation(key_answer, student_answer):
#     model = BertModel.from_pretrained('bert-base-uncased')
#     tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

#     def get_embedding(text):
#         inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True)
#         with torch.no_grad():
#             outputs = model(**inputs)
#         return outputs.last_hidden_state.mean(dim=1).squeeze()

#     key_embedding = get_embedding(key_answer)
#     student_embedding = get_embedding(student_answer)

#     similarity_score = cosine_similarity(key_embedding.unsqueeze(0), student_embedding.unsqueeze(0))
    
#     return similarity_score[0][0]

# def evaluate_student_answer(audio_file, question_prompt):
#     student_answer = audio_to_text(audio_file)
#     print(f"Transcribed Student Answer: {student_answer}")
    
#     if "Could not understand the audio" not in student_answer and "Error in the Google API request" not in student_answer:
#         key_answer = get_key_answer(question_prompt)
#         print(f"Generated Key Answer: {key_answer}")
        
#         similarity_score = bert_similarity_evaluation(key_answer, student_answer)
#         print(f"Similarity Score: {similarity_score}")
        
#         if similarity_score >= 0.60:
#             print("The answer given by the student is correct.")
#         else:
#             print("The answer given by the student is incorrect.")
#     else:
#         print("Transcription failed, unable to evaluate the answer.")

# if __name__ == "__main__":
#     audio_file = 'evaluation model/output.wav' 
#     question_prompt = "What is machine learning?"
#     evaluate_student_answer(audio_file, question_prompt)


# import google.generativeai as genai  # Use the correct import path based on the package

# # Configure the API
# API_KEY = " AIzaSyD9QwMHGOU3daSB-Gldijp70ojpnbcjzgo"
# genai.configure(api_key=API_KEY)

# # List models
# for m in genai.list_models():
#     if 'generateContent' in m.supported_generation_methods:
#         print(m.name)


'''dummy try api'''
# import google.generativeai as genai 
# import speech_recognition as sr
# from transformers import BertModel, BertTokenizer
# import torch
# from sklearn.metrics.pairwise import cosine_similarity
# from pydub import AudioSegment
# from api_k_rotate import get_next_api_key

# # Replace Google API usage
# def convert_to_wav(audio_file_path, output_file_path):
#     audio = AudioSegment.from_file(audio_file_path)
#     audio.export(output_file_path, format='wav')

# def audio_to_text(audio_file):
#     recognizer = sr.Recognizer()
#     with sr.AudioFile(audio_file) as source:
#         audio_data = recognizer.record(source)
#         try:
#             text = recognizer.recognize_google(audio_data)  # You may also use a different ASR
#             return text
#         except sr.UnknownValueError:
#             return "Could not understand the audio."
#         except sr.RequestError:
#             return "Error in the API request."

# # Replace OpenAI logic with Gemini API
# def get_key_answer(prompt):
#     gemini.api_key = get_next_api_key()  # Using rotated API key
#     try:
#         response = gemini.Completion.create(
#             prompt=prompt,
#             max_tokens=100  # adjust as necessary
#         )
#         return response['choices'][0]['text'].strip()  # Assuming Gemini API has a similar response structure
#     except gemini.RateLimitError:
#         print("Rate limit exceeded. Rotating to next API key.")
#         return get_key_answer(prompt)  # Retry with the next key

# # The rest of the code remains the same
# def bert_similarity_evaluation(key_answer, student_answer):
#     model = BertModel.from_pretrained('bert-base-uncased')
#     tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

#     def get_embedding(text):
#         inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True)
#         with torch.no_grad():
#             outputs = model(**inputs)
#         return outputs.last_hidden_state.mean(dim=1).squeeze()

#     key_embedding = get_embedding(key_answer)
#     student_embedding = get_embedding(student_answer)

#     similarity_score = cosine_similarity(key_embedding.unsqueeze(0), student_embedding.unsqueeze(0))
    
#     return similarity_score[0][0]

# def evaluate_student_answer(audio_file, question_prompt):
#     student_answer = audio_to_text(audio_file)
#     print(f"Transcribed Student Answer: {student_answer}")
    
#     if "Could not understand the audio" not in student_answer and "Error in the API request" not in student_answer:
#         key_answer = get_key_answer(question_prompt)
#         print(f"Generated Key Answer: {key_answer}")
        
#         similarity_score = bert_similarity_evaluation(key_answer, student_answer)
#         print(f"Similarity Score: {similarity_score}")
        
#         if similarity_score >= 0.60:
#             print("The answer given by the student is correct.")
#         else:
#             print("The answer given by the student is incorrect.")
#     else:
#         print("Transcription failed, unable to evaluate the answer.")

# if __name__ == "__main__":
#     audio_file = 'evaluation model/output.wav' 
#     question_prompt = "What is machine learning?"
#     evaluate_student_answer(audio_file, question_prompt)
'''this is the code without pydub'''
# import google.generativeai as genai
# import speech_recognition as sr
# import soundfile as sf
# import numpy as np
# from transformers import BertModel, BertTokenizer
# import torch
# from sklearn.metrics.pairwise import cosine_similarity
# from api_k_rotate import get_next_api_key
# import warnings

# def convert_to_wav(audio_file_path, output_file_path):
#     # Read the audio file and save it as WAV
#     data, samplerate = sf.read(audio_file_path)
#     sf.write(output_file_path, data, samplerate)

# def audio_to_text(audio_file):
#     recognizer = sr.Recognizer()
#     with sr.AudioFile(audio_file) as source:
#         audio_data = recognizer.record(source)
#         try:
#             text = recognizer.recognize_google(audio_data)
#             return text
#         except sr.UnknownValueError:
#             return "Could not understand the audio."
#         except sr.RequestError:
#             return "Error in the Google API request."

# def get_key_answer(prompt):
#     genai.configure(api_key=get_next_api_key())
#     try:
#         model = genai.GenerativeModel('gemini-pro')
#         response = model.generate_content(prompt)
#         return response.text.strip()
#     except Exception as e:
#         print(f"Error occurred: {e}. Rotating to next API key.")
#         return get_key_answer(prompt)  # Retry with the next key

# def evaluate_student_answer(audio_file, question_prompt):
#     student_answer = audio_to_text(audio_file)
#     prompt = '''Evaluate the answer on the basis of the question below. 
#     questions : {}
#     answer : {}'''
#     res = get_key_answer(prompt.format(question_prompt, student_answer))
#     return res

# if __name__ == "__main__":
#     audio_file = 'evaluation_model/output.wav' 
#     question_prompt = "What is machine learning?"
#     evaluate_student_answer(audio_file, question_prompt)

# import google.generativeai as genai
# import speech_recognition as sr
# import soundfile as sf
# from api_k_rotate import get_next_api_key
# import warnings

# # Suppress warnings
# warnings.filterwarnings("ignore")

# def convert_to_wav(audio_file_path, output_file_path):
#     data, samplerate = sf.read(audio_file_path)
#     sf.write(output_file_path, data, samplerate, format='WAV')

# def audio_to_text(audio_file):
#     recognizer = sr.Recognizer()
#     with sr.AudioFile(audio_file) as source:
#         audio_data = recognizer.record(source)
#         try:
#             text = recognizer.recognize_google(audio_data)
#             return text
#         except sr.UnknownValueError:
#             return "Could not understand the audio."
#         except sr.RequestError:
#             return "Error in the Google API request."

# def get_key_answer(prompt):
#     genai.configure(api_key=get_next_api_key())
#     try:
#         model = genai.GenerativeModel('gemini-pro')
#         response = model.generate_content(prompt)
#         return response.text.strip()
#     except Exception as e:
#         print(f"Error occurred: {e}. Rotating to next API key.")
#         return get_key_answer(prompt)  # Retry with the next key

# def evaluate_student_answer(audio_file, question_prompt):
#     student_answer = audio_to_text(audio_file)
#     prompt = '''Evaluate the answer based on the question below. 
#     Question: {}
#     Answer: {}
#     Please provide a detailed evaluation of the answer, including its accuracy, completeness, and relevance to the question.'''
#     res = get_key_answer(prompt.format(question_prompt, student_answer))
#     return res

# if __name__ == "__main__":
#     audio_file = 'C:\\Users\\DELL\\Desktop\\api\\output (1).wav' 
#     question_prompt = "What is machine learning?"
#     evaluation_result = evaluate_student_answer(audio_file, question_prompt)
#     print("Evaluation Result:")
#     print(evaluation_result)

'''this code is with gemini api key and the bert similarity'''
# import google.generativeai as genai
# import speech_recognition as sr
# from transformers import BertModel, BertTokenizer
# import torch
# from sklearn.metrics.pairwise import cosine_similarity
# from pydub import AudioSegment
# from api_k_rotate import get_next_api_key

# def convert_to_wav(audio_file_path, output_file_path):
#     audio = AudioSegment.from_file(audio_file_path)
#     audio.export(output_file_path, format='wav')

# def audio_to_text(audio_file):
#     recognizer = sr.Recognizer()
#     with sr.AudioFile(audio_file) as source:
#         audio_data = recognizer.record(source)
#         try:
#             text = recognizer.recognize_google(audio_data)
#             return text
#         except sr.UnknownValueError:
#             return "Could not understand the audio."
#         except sr.RequestError:
#             return "Error in the Google API request."

# def get_key_answer(prompt):
#     genai.configure(api_key=get_next_api_key())
#     try:
#         model = genai.GenerativeModel('gemini-pro')
#         response = model.generate_content(prompt)
#         return response.text.strip()
#     except Exception as e:
#         print(f"Error occurred: {e}. Rotating to next API key.")
#         return get_key_answer(prompt)  # Retry with the next key

# def bert_similarity_evaluation(key_answer, student_answer):
#     model = BertModel.from_pretrained('bert-base-uncased')
#     tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

#     def get_embedding(text):
#         inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True)
#         with torch.no_grad():
#             outputs = model(**inputs)
#         return outputs.last_hidden_state.mean(dim=1).squeeze()

#     key_embedding = get_embedding(key_answer)
#     student_embedding = get_embedding(student_answer)

#     similarity_score = cosine_similarity(key_embedding.unsqueeze(0), student_embedding.unsqueeze(0))
    
#     return similarity_score[0][0]

# def evaluate_student_answer(audio_file, question_prompt):
#     student_answer = audio_to_text(audio_file)
#     print(f"Transcribed Student Answer: {student_answer}")
    
#     if "Could not understand the audio" not in student_answer and "Error in the Google API request" not in student_answer:
#         key_answer = get_key_answer(question_prompt)
#         print(f"Generated Key Answer: {key_answer}")
        
#         similarity_score = bert_similarity_evaluation(key_answer, student_answer)
#         print(f"Similarity Score: {similarity_score}")
        
#         if similarity_score >= 0.60:
#             print("The answer given by the student is correct.")
#         else:
#             print("The answer given by the student is incorrect.")
#     else:
#         print("Transcription failed, unable to evaluate the answer.")

# if __name__ == "__main__":
#     audio_file = 'evaluation model/output.wav' 
#     question_prompt = "What is machine learning?"
#     evaluate_student_answer(audio_file, question_prompt)


'''this code contains the audio input'''
# import google.generativeai as genai
# import speech_recognition as sr
# import soundfile as sf
# from api_k_rotate import get_next_api_key
# import warnings

# # Suppress warnings
# warnings.filterwarnings("ignore")

# # Function to convert audio files to wav format
# def convert_to_wav(audio_file_path, output_file_path):
#     data, samplerate = sf.read(audio_file_path)
#     sf.write(output_file_path, data, samplerate, format='WAV')

# # Function to transcribe audio files to text using Google's Speech Recognition API
# def audio_to_text(audio_file):
#     recognizer = sr.Recognizer()
#     with sr.AudioFile(audio_file) as source:
#         audio_data = recognizer.record(source)
#         try:
#             text = recognizer.recognize_google(audio_data)
#             return text
#         except sr.UnknownValueError:
#             return "Could not understand the audio."
#         except sr.RequestError:
#             return "Error in the Google API request."

# # Function to get the key answer using the Gemini API
# def get_key_answer(prompt):
#     genai.configure(api_key=get_next_api_key())
#     try:
#         model = genai.GenerativeModel('gemini-pro')
#         response = model.generate_content(prompt)
#         return response.text.strip()
#     except Exception as e:
#         print(f"Error occurred: {e}. Rotating to next API key.")
#         return get_key_answer(prompt)  # Retry with the next key

# # Main evaluation function to compare student's answer with question
# def evaluate_student_answer(question_audio_file, student_audio_file):
#     # Transcribe the question audio
#     question_text = audio_to_text(question_audio_file)
#     print(f"Transcribed Question: {question_text}")
    
#     # Transcribe the student's answer audio
#     student_answer = audio_to_text(student_audio_file)
#     print(f"Transcribed Student Answer: {student_answer}")
    
#     # Prompt to send to the generative model
#     prompt = '''Evaluate the answer based on the question below. 
#     Question: {}
#     Answer: {}
#     Please provide how accurate the student is and give if they are right or wrong.'''
    
#     # Get the evaluation from the model using both question and answer
#     evaluation_response = get_key_answer(prompt.format(question_text, student_answer))
#     return evaluation_response

# if __name__ == "__main__":
#     # Provide paths to the question and student answer audio files
#     question_audio_file = 'path_to_question_audio.wav'
#     student_audio_file = 'path_to_student_answer_audio.wav'
    
#     # Call the evaluation function
#     evaluation_result = evaluate_student_answer(question_audio_file, student_audio_file)
#     print("Evaluation Result:")
#     print(evaluation_result)

