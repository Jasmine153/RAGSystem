# import os
# import uuid
import re
import chromadb
import cohere
import pdfplumber
from dotenv import load_dotenv

# READ DATA FROM PDF FILE
def extract_text_from_pdf():
    pdf_path = input("Enter the path to your PDF file: ")
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text


# load_dotenv()
# COHERE_API_KEY = os.getenv("COHERE_API_KEY")
COHERE_API_KEY="API_KEY"
if not COHERE_API_KEY:
    raise ValueError("Cohere API key is missing!")

co = cohere.Client(COHERE_API_KEY)

# CREATE DB INSTANCE
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(name="myPdf")

def split_text_pdf(text, min_chunk_size=600):

    chunks = re.split(r'\n{2,}|\.\s+', text)
    chunks = [chunk.strip() for chunk in chunks if chunk.strip()]
   
    merged_chunks, temp_chunk = [], ""

    for chunk in chunks:
        if len(temp_chunk) + len(chunk) < min_chunk_size:
            temp_chunk += " " + chunk
        else:
            merged_chunks.append(temp_chunk.strip())
            temp_chunk = chunk

    if temp_chunk:
        merged_chunks.append(temp_chunk.strip())

    print(f"Total chunks created: {len(merged_chunks)}")
    return merged_chunks

def get_embeddings(chunks):
    embeddings = co.embed(texts=chunks, model="embed-english-v3.0", input_type="search_document")
    return embeddings.embeddings


def store_embeddings(text):
    chunks = split_text_pdf(text)
    embeddings = get_embeddings(chunks)

    existing_count = collection.count()
    # print(f"Existing documents in ChromaDB: {existing_count}")

    for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
        doc_id = f"doc_{i}"  
        collection.upsert(
            ids=[doc_id],  
            embeddings=[embedding],  
            documents=[chunk]  
        )

    print("Data stored in ChromaDB")


def get_chunks(query,top_k=3):
    embeddings = co.embed(texts=[query], model="embed-english-v3.0", input_type="search_document").embeddings[0]
    results = collection.query(
    query_embeddings=embeddings,
    n_results=top_k,
    )
    retrieved_chunks = results["documents"][0] if results["documents"] else []
    return retrieved_chunks

def get_answer(context, query):
    response = co.generate(
        model='command-xlarge-nightly',
        prompt=f"Context: {context}\n\nQuestion: {query}\n\nAnswer in a clear and concise way:",
        max_tokens=100,
        temperature=0.7
    )
    return response.generations[0].text.strip()

def ask_question():
        query = input("\nEnter your question: ")
        retreived_chunks= get_chunks(query)

        context= "\n".join(retreived_chunks)
        answer= get_answer(query,context)
        print("\nAnswer:", answer)

def main():
    pdf_text = extract_text_from_pdf()
    store_embeddings(pdf_text)
    
    while True:
        ask_question()
        cont = input("\nDo you want to ask another question? (yes/no): ").strip().lower()
        if cont != "yes":
            break


if __name__ == "__main__":
    main()