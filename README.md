# NewTuple-Assignment
Assignment 

Here's an explanation of the approach:

**1.** **Extract Text from PDF Files -** The extract_text_from_pdf function uses the PyPDF2 library to extract text from a PDF file. It iterates over each page of the PDF and concatenates the text from all pages into a single string.

**2. Process the Text -** The process_text function takes the extracted text and splits it into sentences using regular expressions. It removes any leading/trailing whitespace from the sentences and discards empty sentences.

**3. Answer Queries -** The answer_query function takes a query (user question) and a list of sentences as input. It uses the TfidfVectorizer from scikit-learn to convert the sentences and the query into TF-IDF vectors. It computes the cosine similarity between the query vector and each sentence vector. The sentence with the highest cosine similarity to the query is considered the most relevant sentence. The function returns the most relevant sentence, limiting the answer to the first 2-3 lines.

**4. Extract Financial Information -** The extract_financials function uses regular expressions to extract financial information (revenue, profit, and expenses) from the text. The regular expressions are designed to match patterns like "Revenue for the year: $1,234.56 million (up from $1,000.00 million last year)". If a match is found, the function extracts the numerical value, unit, and comparison (if any) for each financial metric. The extracted information is stored in a dictionary for easy access.

The alternate approach is to use Microsoft Azure's Document Intelligence Studio.

https://learn.microsoft.com/en-us/azure/ai-services/document-intelligence/quickstarts/try-document-intelligence-studio?view=doc-intel-4.0.0

https://learn.microsoft.com/en-us/azure/ai-services/document-intelligence/concept-retrieval-augmented-generation?view=doc-intel-4.0.0
