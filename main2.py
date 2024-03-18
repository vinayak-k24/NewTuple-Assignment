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
    tfidf_vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf_vectorizer.fit_transform(sentences + [query])

    # Compute cosine similarity between the query and each sentence
    cosine_similarities = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1]).flatten()

    # Get the most similar sentence
    most_similar_sentence_index = cosine_similarities.argmax()
    most_similar_sentence = sentences[most_similar_sentence_index]

    # Limit the answer to 2-3 lines
    answer_lines = most_similar_sentence.split('. ')[:3]
    answer = '. '.join(answer_lines) + '.'

    return answer

# Step 4: Provide information about financials and comparisons
def extract_financials(text):
    financials = {}

    # Extract revenue information
    revenue_lines = [line for line in text.split('\n') if 'revenue' in line.lower()]
    if revenue_lines:
        revenue_line = revenue_lines[0]
        value_pattern = r'\d+,?\d+\.?\d*'
        value_match = re.search(value_pattern, revenue_line)
        if value_match:
            value = float(value_match.group().replace(',', ''))
            unit = revenue_line.split()[-1]
            financials['revenue'] = {'value': value, 'unit': unit}

    # Extract profit information
    profit_lines = [line for line in text.split('\n') if 'profit' in line.lower()]
    if profit_lines:
        profit_line = profit_lines[0]
        value_pattern = r'\d+,?\d+\.?\d*'
        value_match = re.search(value_pattern, profit_line)
        if value_match:
            value = float(value_match.group().replace(',', ''))
            unit = profit_line.split()[-1]
            financials['profit'] = {'value': value, 'unit': unit}

    # Extract expense information
    expense_lines = [line for line in text.split('\n') if 'expense' in line.lower()]
    if expense_lines:
        expense_line = expense_lines[0]
        value_pattern = r'\d+,?\d+\.?\d*'
        value_match = re.search(value_pattern, expense_line)
        if value_match:
            value = float(value_match.group().replace(',', ''))
            unit = expense_line.split()[-1]
            financials['expenses'] = {'value': value, 'unit': unit}

    return financials

# Define queries to be answered
queries = [
    "What was the key revenue driver for Wipro?",
    "Summarize the chairman's message for TCS",
    "What were the risks outlined by Infosys, and how does that compare?"
]

# Use the functions
directory_path = '/workspaces/NewTuple-Assignment/Assignment_dataset/'

# Loop through all PDF files in the directory
for query in queries:
    print(f"\nQuery: {query}")
    for filename in os.listdir(directory_path):
        if filename.endswith('.pdf'):
            file_path = os.path.join(directory_path, filename)
            try:
                text = extract_text_from_pdf(file_path)
                if text is None:
                    print(f"Failed to extract text from {file_path}")
                    continue

                financials = extract_financials(text)
                company_name = os.path.splitext(filename)[0]
                sentences = process_text(text)

                print(f"\nCompany: {company_name}")
                print(f"Financials: {financials}")

                answer = answer_query(query, sentences)
                print(f"Answer: {answer}")

            except Exception as e:
                print(f"Error processing file {file_path}: {e}")
