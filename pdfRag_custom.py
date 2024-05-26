import PyPDF2
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from transformers import pipeline
from customTokenizer import tokenizer


def extract_text_from_pdf(pdf_path):
    with open(pdf_path, "rb") as file:
        reader = PyPDF2.PdfFileReader(file)
        text = ""
        for page_num in range(reader.numPages):
            page = reader.getPage(page_num)
            text += page.extractText()
    return text


def chunk_text_with_custom_tokenizer(tokenizer, text, max_chunk_size=512):
    tokens = tokenizer.tokenize(text)
    chunks = []
    current_chunk = []
    current_size = 0
    for token in tokens:
        if current_size + len(token) + 1 > max_chunk_size:
            chunks.append(" ".join(current_chunk))
            current_chunk = []
            current_size = 0
        current_chunk.append(token)
        current_size += len(token) + 1
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    return chunks

def retrieve_relevant_chunks(query, top_k=5):
    query_vector = model.encode([query])
    distances, indices = index.search(query_vector, top_k)
    relevant_chunks = [chunks[idx] for idx in indices[0]]
    return relevant_chunks

def answer_question(question, context):
    result = nlp(question=question, context=context)
    return result['answer']


text = extract_text_from_pdf("./example.pdf")
chunks = chunk_text_with_custom_tokenizer(tokenizer, text)

model = SentenceTransformer('all-MiniLM-L6-v2')
chunk_vectors = model.encode(chunks)

index = faiss.IndexFlatL2(chunk_vectors.shape[1])
index.add(np.array(chunk_vectors))

question = "What is the capital of France?"
relevant_chunks = retrieve_relevant_chunks(question)
context = " ".join(relevant_chunks)

nlp = pipeline("question-answering", model="distilbert/distilbert-base-cased-distilled-squad")
answer = answer_question(question, context)
print("Answer:", answer)
