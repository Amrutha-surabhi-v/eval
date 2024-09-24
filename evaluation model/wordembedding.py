import speech_recognition as sr
import numpy as np
from gensim.models import KeyedVectors
from sklearn.metrics.pairwise import cosine_similarity

# Step 1: Convert Audio to Text
def audio_to_text(audio_file_path):
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_file_path) as source:
        audio_data = recognizer.record(source)
    try:
        text = recognizer.recognize_google(audio_data)
        return text
    except sr.UnknownValueError:
        return "Could not understand audio"
    except sr.RequestError as e:
        return f"Error with the request: {e}"

# Step 2: Calculate Word2Vec Embedding Similarity
def get_sentence_embedding(sentence, word_vectors):
    words = sentence.split()
    word_vecs = []
    for word in words:
        if word in word_vectors:
            word_vecs.append(word_vectors[word])
    if len(word_vecs) > 0:
        return np.mean(word_vecs, axis=0)
    else:
        return np.zeros(word_vectors.vector_size)

def evaluate_answer_word2vec(student_answer, correct_answer, word_vectors):
    student_embedding = get_sentence_embedding(student_answer, word_vectors)
    correct_embedding = get_sentence_embedding(correct_answer, word_vectors)
    
    # Compute Cosine Similarity
    similarity_score = cosine_similarity([student_embedding], [correct_embedding])
    return similarity_score[0][0]

# Example usage
student_audio = "C:\\Users\\amrut\\OneDrive\\Desktop\\internship\\evaluation model\\audio_rec.wav"
correct_answer = "Machine learning is a subset of artificial intelligence that allows computers to learn from data and improve their performance on a specific task without being explicitly programmed. Insteadof being hand-coded with rules, the algorithm learns from examples."

# Load pre-trained Word2Vec model (for example, GoogleNews vectors)
word_vectors = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)

# Convert audio to text
student_answer = audio_to_text(student_audio)
print("Student Answer (text):", student_answer)

# Evaluate using Word2Vec
word2vec_similarity_score = evaluate_answer_word2vec(student_answer, correct_answer, word_vectors)
print(f"Word2Vec Cosine Similarity: {word2vec_similarity_score}")
