import PyPDF2
import spacy
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
    doc = nlp(text)
    chunks = [sent.text for sent in doc.sents]
    return chunks

def retrieve_relevant_chunks(query, chunks, top_k=5):
    relevant_chunks = []
    query_doc = nlp(query)
    for chunk in chunks:
        chunk_doc = nlp(chunk)
        similarity = query_doc.similarity(chunk_doc)
        relevant_chunks.append((chunk, similarity))
    relevant_chunks.sort(key=lambda x: x[1], reverse=True)
    return [chunk[0] for chunk in relevant_chunks[:top_k]]

def answer_question(question, context):
    result = nlp(question=question, context=context)
    return result['answer']

pdf_text = extract_text_from_pdf('./example.pdf')
nlp = spacy.load("en_core_web_sm")
chunks = chunk_text(pdf_text)

model = SentenceTransformer('all-MiniLM-L6-v2')
chunk_vectors = model.encode(chunks)

index = faiss.IndexFlatL2(chunk_vectors.shape[1])
index.add(np.array(chunk_vectors))

query = "What year did the French Revolution take place?"
relevant_chunks = retrieve_relevant_chunks(query, chunks)
context = " ".join(relevant_chunks)

nlp = pipeline("question-answering", model="distilbert/distilbert-base-cased-distilled-squad")
answer = answer_question(query, context)
print("Answer:", answer)
