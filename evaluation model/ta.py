# from transformers import T5Tokenizer, T5ForConditionalGeneration

# # Load pre-trained T5 model and tokenizer
# model = T5ForConditionalGeneration.from_pretrained('t5-base')
# tokenizer = T5Tokenizer.from_pretrained('t5-base')

# # Function to perform T5-based question answering
# def t5_qa(question, context):
#     input_text = f"question: {question}  context: {context}"
#     input_ids = tokenizer(input_text, return_tensors='pt').input_ids
#     output = model.generate(input_ids)
#     answer = tokenizer.decode(output[0], skip_special_tokens=True)
#     return answer

# # Example usage
# question = "What is machine learning?"
# context = ("Machine learning is a branch of artificial intelligence (AI) focused on building applications that learn "
#            "from data and improve their accuracy over time without being programmed to do so. In data science, an "
#            "algorithm is trained on data to find patterns or make predictions.")
# answer = t5_qa(question, context)
# print(f"Answer: {answer}")


from transformers import T5Tokenizer, T5ForConditionalGeneration, BertModel, BertTokenizer
import torch
from sklearn.metrics.pairwise import cosine_similarity
import speech_recognition as sr

def t5_qa(question, context):
    """Performs T5-based question answering."""
    input_text = f"question: {question} context: {context}"
    input_ids = tokenizer(input_text, return_tensors='pt').input_ids
    output = model.generate(input_ids)
    answer = tokenizer.decode(output[0], skip_special_tokens=True)
    return answer

def bert_similarity_evaluation(key_answer, student_answer):
    """Evaluates similarity between two texts using BERT."""
    # Load pre-trained BERT model and tokenizer
    model = BertModel.from_pretrained('bert-base-uncased')
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # Tokenize input texts and get embeddings
    def get_embedding(text):
        inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True)
        with torch.no_grad():
            outputs = model(**inputs)
        return outputs.last_hidden_state.mean(dim=1).squeeze()

    # Get embeddings for both answers
    key_embedding = get_embedding(key_answer)
    student_embedding = get_embedding(student_answer)

    # Compute cosine similarity
    similarity_score = cosine_similarity(key_embedding.unsqueeze(0), student_embedding.unsqueeze(0))
    return similarity_score[0][0]

def audio_to_text(audio_file):
    """Transcribes audio to text."""
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

def evaluate_student_answer(audio_file, key_answer):
    """Evaluates a student's answer based on audio input and BERT similarity."""
    # Convert audio to text
    student_answer = audio_to_text(audio_file)
    print(f"Transcribed Student Answer: {student_answer}")

    # If the transcription was successful
    if "Could not understand the audio" not in student_answer and "Error in the Google API request" not in student_answer:
        # Perform T5-based question answering if needed
        # answer = t5_qa(question, context)  # Uncomment if T5 is required

        # Perform BERT similarity evaluation
        similarity_score = bert_similarity_evaluation(key_answer, student_answer)
        print(f"Similarity Score: {similarity_score}")

        # Check if similarity is above a threshold (e.g., 60%)
        if similarity_score >= 0.60:
            print("The answer given by the student is correct.")
        else:
            print("The answer given by the student is incorrect.")
    else:
        print("Transcription failed, unable to evaluate the answer.")

# Load pre-trained T5 model and tokenizer
model = T5ForConditionalGeneration.from_pretrained('t5-base')
tokenizer = T5Tokenizer.from_pretrained('t5-base')

# Example usage
audio_file = 'C:\\Users\\amrut\\OneDrive\\Desktop\\evaluation model\\agiudio_rec.mp3'  # Replace with the path to the audio file
key_answer = "Machine learning is a subset of artificial intelligence that allows computers to learn from data and improve their performance on a specific task without being explicitly programmed. Insteadof being hand-coded with rules, the algorithm learns from examples."

evaluate_student_answer(audio_file, key_answer)
