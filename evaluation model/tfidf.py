import speech_recognition as sr
from sklearn.feature_extraction.text import TfidfVectorizer
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

# Step 2: Calculate TF-IDF and Cosine Similarity
def evaluate_answer_tfidf(student_answer, correct_answer):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([student_answer, correct_answer])
    
    # Compute Cosine Similarity between Student's Answer and Correct Answer
    similarity_score = cosine_similarity(tfidf_matrix[0], tfidf_matrix[1])
    return similarity_score[0][0]

# Example usage
student_audio = "C:\\Users\\amrut\\OneDrive\\Desktop\\internship\\evaluation model\\audio_rec.wav"
correct_answer = "Machine learning is a subset of artificial intelligence that allows computers to learn from data and improve their performance on a specific task without being explicitly programmed. Insteadof being hand-coded with rules, the algorithm learns from examples."

# Convert audio to text
student_answer = audio_to_text(student_audio)
print("Student Answer (text):", student_answer)

# Evaluate using TF-IDF
tfidf_similarity_score = evaluate_answer_tfidf(student_answer, correct_answer)
print(f"TF-IDF Cosine Similarity: {tfidf_similarity_score}")
