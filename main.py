from PyPDF2 import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
import os

# Step 1: Extract text from PDF (modified)
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

# Step 3: Answer queries based on the processed text (moved outside loop)
def answer_query(query, all_sentences, doc_ids):
    tfidf_vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf_vectorizer.fit_transform(all_sentences + [query])

    cosine_similarities = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1]).flatten()

    # Identify top scoring documents (e.g., top 3)
    top_doc_indices = cosine_similarities.argsort()[-3:]
    top_doc_scores = cosine_similarities[top_doc_indices]

    # Find most similar sentence within top documents
    most_similar_idx = None
    highest_score = 0
    for i, (score, doc_id) in enumerate(zip(top_doc_scores, doc_ids[top_doc_indices])):
        if score > highest_score and doc_id in doc_ids:  # Filter by relevant documents
            highest_score = score
            most_similar_idx = top_doc_indices[i]

    return all_sentences[most_similar_idx] if most_similar_idx is not None else "No relevant answer found."


# Step 4: Provide information about financials and comparisons
def extract_financials(text):
    financials = {}

    # Extract revenue information
    revenue_pattern = r'Revenue\s*[\(\w\s]*:\s*(\d+\.\d+)\s*(\w+)\s*([\(\w\s]*\d+\.\d+\s*\w+)*'
    revenue_matches = re.findall(revenue_pattern, text)
    if revenue_matches:
        revenue_value, revenue_unit, revenue_comparison = revenue_matches[0]
        financials['revenue'] = {
            'value': float(revenue_value),
            'unit': revenue_unit,
            'comparison': revenue_comparison.strip() if revenue_comparison else None
        }

    # Extract profit information
    profit_pattern = r'Profit\s*[\(\w\s]*:\s*(\d+\.\d+)\s*(\w+)\s*([\(\w\s]*\d+\.\d+\s*\w+)*'
    profit_matches = re.findall(profit_pattern, text)
    if profit_matches:
        profit_value, profit_unit, profit_comparison = profit_matches[0]
        financials['profit'] = {
            'value': float(profit_value),
            'unit': profit_unit,
            'comparison': profit_comparison.strip() if profit_comparison else None
        }

    # Extract expense information
    expense_pattern = r'Expenses\s*[\(\w\s]*:\s*(\d+\.\d+)\s*(\w+)\s*([\(\w\s]*\d+\.\d+\s*\w+)*'
    expense_matches = re.findall(expense_pattern, text)
    if expense_matches:
        expense_value, expense_unit, expense_comparison = expense_matches[0]
        financials['expenses'] = {
            'value': float(expense_value),
            'unit': expense_unit,
            'comparison': expense_comparison.strip() if expense_comparison else None
        }

    return financials



# Use the functions
directory = '/workspaces/NewTuple-Assignment/Assignment_dataset'
file_paths = []
all_sentences = []  # Store sentences from all PDFs

for filename in os.listdir(directory):
    if filename.endswith('.pdf'):
        file_path = os.path.join(directory, filename)
        file_paths.append(file_path)

        text = extract_text_from_pdf(file_path)
        sentences = process_text(text)
        all_sentences.extend(sentences)  # Add sentences to combined list

for query in ["What was the key revenue driver for Wipro in the year 2022?",
              "Summarize the chairman's message for TCS in 2023",
              "What were the risks outlined by Infosys in 2021, and how does that compare with 2022?"]:
    answer = answer_query(query, all_sentences)
    print(f"\nFor query: '{query}'")
    print(f"Answer: {answer}")
