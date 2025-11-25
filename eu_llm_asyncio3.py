CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE documents (
    id SERIAL PRIMARY KEY,
    content TEXT,
    tax_type VARCHAR(50),
    processed BOOLEAN DEFAULT FALSE
);

CREATE TABLE tax_analysis_results (
    id SERIAL PRIMARY KEY,
    doc_id INTEGER REFERENCES documents(id),
    chunk_index INTEGER,
    question TEXT,
    answer TEXT,
    embedding_vector vector(384) -- 384 dimensions for all-MiniLM-L6-v2
);

import asyncio
import time
import os
from typing import List, Dict, Any
import psycopg
from pgvector.psycopg import register_vector
from langchain_community.chat_models import ChatOllama
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate

import asyncio
import time
import os
from typing import List, Dict, Any
import psycopg
from pgvector.psycopg import register_vector
from langchain_community.chat_models import ChatOllama
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate


import asyncio
import time
import os
from typing import List, Dict, Any
import psycopg
from pgvector.psycopg import register_vector
from langchain_community.chat_models import ChatOllama
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate

# --- CONFIGURATION ---
DB_CONN_STRING = "postgresql://user:password@localhost:5432/tax_db"
OLLAMA_MODEL = "bielik"  # Or "plumm"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2" # Or a Polish specific model like 'sdadas/st-polish-paraphrase-from-distilroberta'
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
MAX_CONCURRENT_LLM_REQUESTS = 4 # Adjust based on your GPU VRAM

# --- PERFORMANCE TIMER CONTEXT MANAGER ---
class StepTimer:
    def __init__(self, step_name):
        self.step_name = step_name
        self.start_time = 0

    def __enter__(self):
        print(f"--- Starting: {self.step_name} ---")
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed = time.perf_counter() - self.start_time
        print(f"--- Finished: {self.step_name} in {elapsed:.4f} seconds ---\n")

# --- TAX LOGIC ---
def get_questions_for_tax_type(tax_type: str) -> List[str]:
    """Returns specific questions based on the tax context."""
    prompts = {
        "VAT": [
            "Jaka jest stawka VAT dla opisanej transakcji?",
            "Kiedy powstaje obowiązek podatkowy w tym przypadku?",
            "Czy transakcja korzysta ze zwolnienia z VAT?",
            "Kto jest podatnikiem zobowiązanym do rozliczenia?",
            "Czy przysługuje prawo do odliczenia podatku naliczonego?",
            "Jakie jest miejsce świadczenia dla celów VAT?",
            "Czy wymagane jest wystawienie faktury?",
            "Czy transakcja podlega mechanizmowi podzielonej płatności?",
            "Czy jest to wewnątrzwspólnotowa dostawa lub nabycie?",
            "Jakie obowiązki ewidencyjne wiążą się z tą czynnością?"
        ],
        "CIT": [
            "Czy wydatek stanowi koszt uzyskania przychodu?",
            "Jaka jest data powstania przychodu należnego?",
            "Czy przychód jest zwolniony z opodatkowania?",
            "Jaka stawka CIT ma zastosowanie (9% czy 19%)?",
            "Czy stosuje się przepisy o cenach transferowych?",
            "Czy powstają różnice kursowe podatkowe?",
            "Czy transakcja podlega podatkowi u źródła (WHT)?",
            "Czy wydatek jest wyłączony z kosztów na podstawie art. 16 ust. 1?",
            "Czy konieczna jest korekta kosztów lub przychodów?",
            "Do jakiego źródła przychodów (zyski kapitałowe/inne) należy przypisać kwotę?"
        ],
        "PIT": [
            "Do jakiego źródła przychodów zalicza się to świadczenie?",
            "Czy dochód jest zwolniony z PIT (np. art. 21)?",
            "Czy przysługują zryczałtowane koszty uzyskania przychodu?",
            "Jaka jest podstawa obliczenia zaliczki na podatek?",
            "Czy stosuje się skalę podatkową czy podatek liniowy?",
            "Czy podatnik może skorzystać z ulg (np. na dzieci, internet)?",
            "Czy świadczenie podlega oskładkowaniu ZUS?",
            "Kto pełni funkcję płatnika w tym przypadku?",
            "Kiedy należy wykazać przychód w zeznaniu rocznym?",
            "Czy transakcja wymaga prowadzenia ewidencji przychodów?"
        ],
    }
    # Return 10 generic questions if tax_type is unknown to maintain consistency
    fallback = ["Proszę streścić jeden z kluczowych aspektów podatkowych tego tekstu."] * 10
    return prompts.get(tax_type, fallback)

# --- RAG RETRIEVAL & GENERATION ---

async def retrieve_relevant_data(
    query: str, 
    limit: int = 5,
    similarity_threshold: float = 0.5
) -> List[Dict[str, Any]]:
    """
    RAG Step 1 (Selection): Embeds the query and selects relevant data from DB.
    """
    print(f"\n--- RAG Selection: Searching for '{query}' ---")
    
    # We re-initialize the embedding model here for standalone usage
    embeddings_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    
    # Generate vector for the user's query
    loop = asyncio.get_running_loop()
    query_vector = await loop.run_in_executor(
        None, 
        embeddings_model.embed_query, 
        query
    )

    async with await psycopg.AsyncConnection.connect(DB_CONN_STRING) as aconn:
        await register_vector(aconn)
        async with aconn.cursor() as cur:
            # PGVector query: <=> is cosine distance. 
            # We select rows with lowest distance (highest similarity).
            search_sql = """
            SELECT doc_id, question, answer, (1 - (embedding_vector <=> %s)) as similarity
            FROM tax_analysis_results
            WHERE 1 - (embedding_vector <=> %s) > %s
            ORDER BY embedding_vector <=> %s ASC
            LIMIT %s
            """
            
            await cur.execute(search_sql, (query_vector, query_vector, similarity_threshold, query_vector, limit))
            rows = await cur.fetchall()
            
            results = []
            for doc_id, q, ans, sim in rows:
                results.append({
                    "doc_id": doc_id,
                    "original_question": q,
                    "answer_content": ans,
                    "similarity": sim
                })
            
            print(f"Found {len(results)} relevant records.")
            return results

async def answer_user_query_rag(user_query: str):
    """
    RAG Step 2 (Generation): Answers the user query using selected data.
    """
    # 1. Select Data
    context_docs = await retrieve_relevant_data(user_query)
    
    if not context_docs:
        print("No relevant data found in knowledge base.")
        return

    # 2. Construct Context String
    context_text = "\n\n".join(
        [f"Info z dokumentu {d['doc_id']} (Pytanie: {d['original_question']}): {d['answer_content']}" 
         for d in context_docs]
    )

    # 3. Generate Answer with LLM
    print("Generating final answer...")
    llm = ChatOllama(model=OLLAMA_MODEL, temperature=0)
    
    prompt_template = ChatPromptTemplate.from_template(
        """Jesteś ekspertem podatkowym. Odpowiedz na pytanie użytkownika WYŁĄCZNIE na podstawie poniższych fragmentów wiedzy.
        
        Baza Wiedzy:
        {context}
        
        Pytanie Użytkownika: {question}
        
        Odpowiedź:"""
    )
    
    chain = prompt_template | llm
    response = await chain.ainvoke({"context": context_text, "question": user_query})
    
    print("\n=== FINAL RAG ANSWER ===")
    print(response.content)
    print("========================\n")

# --- ASYNC LLM WORKER ---
async def generate_answer(
    chunk_text: str, 
    question: str, 
    llm: ChatOllama, 
    semaphore: asyncio.Semaphore
) -> Dict[str, Any]:
    """
    Generates an answer for a specific chunk and question.
    Uses a semaphore to limit concurrent calls to the local Ollama instance.
    """
    async with semaphore:
        prompt = ChatPromptTemplate.from_template(
            "Kontekst:\n{context}\n\nPytanie: {question}\n\nOdpowiedź (zwięźle):"
        )
        chain = prompt | llm
        
        try:
            # Run the chain asynchronously
            response = await chain.ainvoke({"context": chunk_text, "question": question})
            return {
                "question": question,
                "answer": response.content,
                "chunk_text": chunk_text,
                "success": True
            }
        except Exception as e:
            print(f"Error generating answer: {e}")
            return {"success": False}

# --- MAIN PIPELINE ---
async def process_tax_documents():
    
    # Initialize Models
    print("Initializing Models...")
    llm = ChatOllama(model=OLLAMA_MODEL, temperature=0)
    
    # We load embeddings synchronously here as usually Init is fast, 
    # but actual embedding generation will be run in a thread executor or batch
    embeddings_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    
    # Text Splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, 
        chunk_overlap=CHUNK_OVERLAP
    )

    # Semaphore to prevent melting the GPU
    sem = asyncio.Semaphore(MAX_CONCURRENT_LLM_REQUESTS)

    async with await psycopg.AsyncConnection.connect(DB_CONN_STRING, autocommit=True) as aconn:
        # Register pgvector type handling for this connection
        await register_vector(aconn)
        
        async with aconn.cursor() as cur:
            
            # --- STEP 1: DOWNLOAD BATCH ---
            with StepTimer("1. Fetch Data from Postgres"):
                # Assuming table 'documents' has columns: id, content, tax_type
                await cur.execute(
                    "SELECT id, content, tax_type FROM documents WHERE processed = FALSE LIMIT 10"
                )
                rows = await cur.fetchall()
                if not rows:
                    print("No documents to process.")
                    return

            processing_tasks = []
            
            # --- STEP 2 & 3: ASSIGN PROMPTS AND SPLIT CHUNKS ---
            with StepTimer("2 & 3. Assign Prompts & Split Text"):
                for row_id, content, tax_type in rows:
                    # 2. Assign Prompt/Questions
                    questions = get_questions_for_tax_type(tax_type)
                    
                    # 3. Split Text
                    chunks = text_splitter.split_text(content)
                    
                    for chunk_idx, chunk in enumerate(chunks):
                        for question in questions:
                            # Prepare task metadata
                            task_meta = {
                                "doc_id": row_id,
                                "chunk_idx": chunk_idx,
                                "question": question,
                                "chunk_text": chunk
                            }
                            processing_tasks.append(task_meta)

            print(f"Prepared {len(processing_tasks)} sub-tasks (Chunks * Questions).")

            # --- STEP 4: GENERATE ANSWERS (LLM) ---
            generated_results = []
            with StepTimer("4. LLM Answer Generation"):
                # Create async tasks for all chunks/questions
                tasks = [
                    generate_answer(t["chunk_text"], t["question"], llm, sem)
                    for t in processing_tasks
                ]
                
                # Run them concurrently (limited by semaphore)
                results = await asyncio.gather(*tasks)
                
                # Merge results with metadata
                for i, res in enumerate(results):
                    if res["success"]:
                        combined = processing_tasks[i].copy()
                        combined["answer"] = res["answer"]
                        generated_results.append(combined)

            if not generated_results:
                print("No answers generated.")
                return

            # --- STEP 5: VECTORIZATION ---
            with StepTimer("5. Vectorization of Answers"):
                # We vectorize the ANSWER, as requested.
                # Extract just the text to embed
                texts_to_embed = [item["answer"] for item in generated_results]
                
                # Embeddings are usually CPU intensive or sync in LangChain wrapper.
                # Run in a separate thread to avoid blocking the asyncio loop.
                loop = asyncio.get_running_loop()
                vectors = await loop.run_in_executor(
                    None, 
                    embeddings_model.embed_documents, 
                    texts_to_embed
                )
                
                # Attach vectors back to results
                for i, vector in enumerate(vectors):
                    generated_results[i]["vector"] = vector

            # --- STEP 6: SAVE TO POSTGRES ---
            with StepTimer("6. Save Results to DB"):
                # Assuming table: tax_analysis_results (doc_id, chunk_index, question, answer, embedding_vector)
                insert_query = """
                INSERT INTO tax_analysis_results 
                (doc_id, chunk_index, question, answer, embedding_vector)
                VALUES (%s, %s, %s, %s, %s)
                """
                
                data_to_insert = [
                    (
                        r["doc_id"], 
                        r["chunk_idx"], 
                        r["question"], 
                        r["answer"], 
                        r["vector"]
                    )
                    for r in generated_results
                ]
                
                await cur.executemany(insert_query, data_to_insert)
                
                # Mark source docs as processed
                processed_ids = list(set(r["doc_id"] for r in rows))
                await cur.execute(
                    "UPDATE documents SET processed = TRUE WHERE id = ANY(%s)",
                    (processed_ids,)
                )

if __name__ == "__main__":
    try:
        asyncio.run(process_tax_documents())
    except KeyboardInterrupt:
        print("Process stopped by user.")

    # Example of how to use the new RAG function (uncomment to test):
    # asyncio.run(answer_user_query_rag("Jakie są stawki VAT?"))
