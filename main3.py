import os
from PyPDF2 import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re

# Step 1: Extract text from PDF
def extract_text_from_pdf(file_path):
    try:
        pdf_reader = PdfReader(file_path)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        return text
    except Exception as e:
        print(f"Error extracting text from {file_path}: {e}")
        return None

# Step 2: Process the text (split into sentences and remove special characters)
def process_text(text):
    if text is None:
        return []
    sentences = re.split(r'[.?!]', text)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 0]
    return sentences

# Step 3: Answer queries based on the processed text
def answer_query(query, sentences):
    relevant_sentences = []
    for sentence in sentences:
        if query.lower() in sentence.lower():
            relevant_sentences.append(sentence)

    if not relevant_sentences:
        return "No relevant information found."

    tfidf_vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf_vectorizer.fit_transform(relevant_sentences + [query])

    # Compute cosine similarity between the query and each sentence
    cosine_similarities = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1]).flatten()

    # Get the most similar sentence
    most_similar_sentence_index = cosine_similarities.argmax()
    most_similar_sentence = relevant_sentences[most_similar_sentence_index]

    # Limit the answer to 2-3 lines
    answer_lines = most_similar_sentence.split('. ')[:3]
    answer = '. '.join(answer_lines) + '.'

    return answer

# Use the functions
directory_path = '/workspaces/NewTuple-Assignment/Assignment_dataset/'

# Define queries to be answered
queries = [
    "What was the key revenue driver?",
    "Summarize the chairman's message",
    "What were the risks outlined, and how does that compare?"
]

# Answer queries for each file
for query in queries:
    print(f"\nQuery: {query}")
    relevant_sentences = []
    for filename in os.listdir(directory_path):
        if filename.endswith('.pdf'):
            file_path = os.path.join(directory_path, filename)

            text = extract_text_from_pdf(file_path)
            if text is None:
                print(f"Failed to extract text from {file_path}")
                continue

            sentences = process_text(text)
            relevant_sentences.extend(sentences)

    answer = answer_query(query, relevant_sentences)
    print(f"Answer: {answer}")
