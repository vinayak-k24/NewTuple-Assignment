import os
from PyPDF2 import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re

# Step 1: Extract text from PDF
def extract_text_from_pdf(file_path):
    pdf_reader = PdfReader(file_path)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

# Step 2: Process the text (split into sentences and remove special characters)
def process_text(text):
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

# Step 5: Extract year from text
def extract_year(text):
    first_sentence = process_text(text)[0]
    words = first_sentence.split()
    for word in words:
        if word.isdigit() and len(word) == 4:
            return int(word)
    return None

# Define queries to be answered
queries = [
    "What was the key revenue driver for Wipro in the year 2022?",
    "Summarize the chairman's message for TCS in 2023",
    "What were the risks outlined by Infosys in 2021, and how does that compare with 2022?"
]

# Use the functions
directory_path = '/workspaces/NewTuple-Assignment/Assignment_dataset/'

# Loop through all PDF files in the directory
for filename in os.listdir(directory_path):
    if filename.endswith('.pdf'):
        file_path = os.path.join(directory_path, filename)
        try:
            text = extract_text_from_pdf(file_path)
            financials = extract_financials(text)
            company_name = os.path.splitext(filename)[0]
            year = extract_year(text)
            sentences = process_text(text)

            # Skip files without valid year
            if year is None:
                continue

            print(f"\nCompany: {company_name}, Year: {year}")
            print(f"Financials: {financials}")

            for query in queries:
                answer = answer_query(query, sentences)
                print(f"\nQuery: {query}")
                print(f"Answer: {answer}")

        except Exception as e:
            print(f"Error processing file {file_path}: {e}")