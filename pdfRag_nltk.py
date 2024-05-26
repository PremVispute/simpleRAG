import PyPDF2
from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from transformers import pipeline
def extract_text_from_pdf(pdf_path):
    pdf_reader = PyPDF2.PdfFileReader(pdf_path)
    text = ""
    for page_num in range(pdf_reader.numPages):
        page = pdf_reader.getPage(page_num)
        text += page.extract_text()
    return text

def chunk_text(text):
    sentences = sent_tokenize(text)
    return sentences

def retrieve_relevant_chunks(query, top_k=5):
    query_vector = model.encode([query])
    distances, indices = index.search(query_vector, top_k)
    relevant_chunks = [chunks[idx] for idx in indices[0]]
    return relevant_chunks

def answer_question(question, context):
    result = nlp(question=question, context=context)
    return result['answer']

pdf_text = extract_text_from_pdf('./example.pdf')
chunks = chunk_text(pdf_text)

model = SentenceTransformer('all-MiniLM-L6-v2')
chunk_vectors = model.encode(chunks)

index = faiss.IndexFlatL2(chunk_vectors.shape[1])
index.add(np.array(chunk_vectors))

query = "What year did the French Revolution take place?"
relevant_chunks = retrieve_relevant_chunks(query)
context = " ".join(relevant_chunks)

nlp = pipeline("question-answering", model="distilbert/distilbert-base-cased-distilled-squad")
answer = answer_question(query, context)
print("Answer:", answer)
