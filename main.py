import os
import csv
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

# # Step 4: Provide information about financials and comparisons
# def extract_financials(text):
#     financials = {}

#     # Extract revenue information
#     revenue_pattern = r'Revenue\s*[\(\w\s]*?:\s*(\d+\.\d+)\s*(\w+)\s*([\(\w\s]*?\d+\.\d+\s*\w+)*'
#     revenue_matches = re.findall(revenue_pattern, text)
#     if revenue_matches:
#         revenue_value, revenue_unit, revenue_comparison = revenue_matches[0]
#         financials['revenue'] = {
#             'value': float(revenue_value),
#             'unit': revenue_unit,
#             'comparison': revenue_comparison.strip() if revenue_comparison else None
#         }

#     # Extract profit information
#     profit_pattern = r'Profit\s*[\(\w\s]*?:\s*(\d+\.\d+)\s*(\w+)\s*([\(\w\s]*?\d+\.\d+\s*\w+)*'
#     profit_matches = re.findall(profit_pattern, text)
#     if profit_matches:
#         profit_value, profit_unit, profit_comparison = profit_matches[0]
#         financials['profit'] = {
#             'value': float(profit_value),
#             'unit': profit_unit,
#             'comparison': profit_comparison.strip() if profit_comparison else None
#         }

#     # Extract expense information
#     expense_pattern = r'Expenses\s*[\(\w\s]*?:\s*(\d+\.\d+)\s*(\w+)\s*([\(\w\s]*?\d+\.\d+\s*\w+)*'
#     expense_matches = re.findall(expense_pattern, text)
#     if expense_matches:
#         expense_value, expense_unit, expense_comparison = expense_matches[0]
#         financials['expenses'] = {
#             'value': float(expense_value),
#             'unit': expense_unit,
#             'comparison': expense_comparison.strip() if expense_comparison else None
#         }

#     return financials

def extract_financials(text):
    financials = {}

    # Extract revenue, profit, and expense information
    for key in ['revenue', 'profit', 'expenses']:
        pattern = fr'{key.capitalize()}\s*:\s*(\d+\.?\d*)\s*(\w+)?'
        matches = re.findall(pattern, text, re.IGNORECASE)
        if matches:
            value, unit = matches[0]
            financials[key] = {
                'value': float(value),
                'unit': unit or None,
            }

    return financials

# Step 5: Extract year from text
def extract_year(text):
    first_sentence = process_text(text)[0]
    words = first_sentence.split()
    for word in words:
        if word.isdigit() and len(word) == 4:
            return int(word)
    return None

# Use the functions
directory_path = '/workspaces/NewTuple-Assignment/Assignment_dataset/'
output_file = 'financial_reports.csv'

# Create a CSV file and write the header row
fieldnames = ['company_name', 'year', 'revenue_value', 'revenue_unit', 'revenue_comparison', 'profit_value', 'profit_unit', 'profit_comparison', 'expense_value', 'expense_unit', 'expense_comparison']
with open(output_file, 'w', newline='') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()

# Loop through all PDF files in the directory
for filename in os.listdir(directory_path):
    if filename.endswith('.pdf'):
        print(f"Processing file: {filename}")
        file_path = os.path.join(directory_path, filename)
        text = extract_text_from_pdf(file_path)
        print(f"Extracted text: {text[:100]}...")  # Print the first 100 characters of the text
        financials = extract_financials(text)
        print(f"Extracted financials: {financials}")
        company_name = os.path.splitext(filename)[0]
        year = extract_year(text)
        print(f"Extracted year: {year}")
        sentences = process_text(text)

        # Write financial information to the CSV file
        row = {
            'company_name': company_name,
            'year': year,
            'revenue_value': financials.get('revenue', {}).get('value', ''),
            'revenue_unit': financials.get('revenue', {}).get('unit', ''),
            'revenue_comparison': financials.get('revenue', {}).get('comparison', ''),
            'profit_value': financials.get('profit', {}).get('value', ''),
            'profit_unit': financials.get('profit', {}).get('unit', ''),
            'profit_comparison': financials.get('profit', {}).get('comparison', ''),
            'expense_value': financials.get('expenses', {}).get('value', ''),
            'expense_unit': financials.get('expenses', {}).get('unit', ''),
            'expense_comparison': financials.get('expenses', {}).get('comparison', '')
        }
        writer.writerow(row)

        # Answer example queries if they are relevant to the PDF file
        queries = [
            "What was the key revenue driver for Wipro in the year 2022?",
            "Summarize the chairman's message for TCS in 2023",
            "What were the risks outlined by Infosys in 2021, and how does that compare with 2022?"
        ]
        for query in queries:
            query_company_name = query.split(' ')[-1]
            query_year = int(query.split(' ')[-3])
            if query_company_name == company_name and query_year == year:
                answer = answer_query(query, sentences)
                print(f"\nQuery: {query}")
                print(f"Answer for {company_name} ({year}): {answer}")
                with open('output.txt', 'a') as f:
                    f.write(f"\nQuery: {query}")
                    f.write(f"\nAnswer for {company_name} ({year}): {answer}")

print(f"\nFinancial reports data has been written to {output_file}")
print("Output has been written to output.txt")