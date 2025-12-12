from langchain.embeddings import OllamaEmbeddings
from langchain.vectorstores.pgvector import PGVector
from langchain.llms import Ollama
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.chains.retrieval_qa.base import RetrievalQA

# Prompt template in Polish
prompt_template = """
Jesteś ekspertem w dziedzinie prawa podatkowego w obszarze {typ_podatku}, 
specjalizującym się w analizie interpretacji indywidualnych. Z wprowadzonej treści {tresc} należy wykonać następujące sprawdzenia:
    Ustal, o jaki podatek chodzi.
    Jeśli wskazano konkretny przepis prawa podatkowego, zwróć szczególną uwagę na interpretacje, które również go zawierają.
    Jeżeli zapytanie dotyczy podatku CIT, podczas analizy porównawczej interpretacji, szczególną uwagę zwróć na następujące aspekty:
    • Czy wnioskodawca podlega opodatkowaniu podatkiem dochodowym od osób prawnych?
    • Czy wnioskodawca jest osobą fizyczną czy osobą prawną?
    • Jaka jest forma prawna wnioskodawcy?
    • Czy wnioskodawca podlega w Polsce obowiązkowi podatkowemu od całości swoich dochodów bez względu na miejsce ich osiągania, czy tylko od dochodów osiąganych w Polsce?
    • Czy wnioskodawca występuje w charakterze podatnika czy płatnika?
    • Czy została wskazana forma opodatkowania dochodów/przychodów?
    • Czy wnioskodawca korzysta z ulg podatkowych?
    • Jaki jest zakres działalności prowadzonej przez wnioskodawcę?

Zwróć wyniki w postaci słownika.
"""

# Prompt setup
prompt = PromptTemplate(
    input_variables=["typ_podatku", "tresc"],
    template=prompt_template
)

# Initialize local Ollama model
llm = Ollama(model="llama3")

# Create the chain
chain = LLMChain(llm=llm, prompt=prompt)

# Embedding model
embedding = OllamaEmbeddings(model="llama3")

# PostgreSQL connection string (adjust credentials)
CONNECTION_STRING = "postgresql+psycopg://username:password@localhost:5432/dbname"

# Vector store
vectorstore = PGVector(
    connection_string=CONNECTION_STRING,
    embedding_function=embedding,
    collection_name="tax_law_embeddings"
)

# Retrieval QA chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever(),
    return_source_documents=True
)

# Example query
user_query = {
    "typ_podatku": "CIT",
    "tresc": "Czy spółka z o.o. prowadząca działalność badawczo-rozwojową może skorzystać z ulgi IP Box?"
}

# Run the prompt
response = chain.run(user_query)

# Similarity search
similar_docs = vectorstore.similarity_search(user_query["tresc"], k=5)

# Output
print("Structured Analysis:\n", response)
print("\nTop 5 Similar Documents:")
for doc in similar_docs:
    print(doc.page_content)








#     test_plumm1()


#     query = "czym jest podatek?"
#     ollama_model = load_ollama_model(model_name="PRIHLOP/PLLuM:8b")
#     prompt = build_rag_prompt(query)
#     generated_answer = ollama_model(prompt)
#     print(generated_answer)


#     ef load_ollama_model(model_name: str) -> Callable[[str], str]:
#     import ollama
#     def real_ollama_model(prompt: str) -> str:
#         response = ollama.chat(model=model_name, messages=[
#             {'role': 'user', 'content': prompt}
#         ])
#         return response['message']['content']
#     return real_ollama_model


# def test_plumm1():
#     messages = [
#         ("system",
#         "You are a helpful assistant that translates English to French. Translate the user sentence.",
#         ),
#         ("question", "I love programming."),
#     ]


#     template = """Question: {question}

#     Answer: Pomyslę i odpowiem po polsku."""

#     prompt = ChatPromptTemplate.from_template(template)
#     print(prompt)

#     # model = OllamaLLM(model="PRIHLOP/PLLuM:8b", temperature=0.9, max_tokens=1024)

#     # chain = prompt | model

#     # response = chain.invoke({"question": "Jakie są rodzaje podatków w Polsce?"})




#     #print(response)


# def build_rag_prompt(query: str) -> str:
#     """
#     Helper function to build a prompt for the RAG model.
#     """
#     #context = "\n\n".join(context_chunks)

#     prompt = f"""
#     Użyj poniższego kontekstu, aby odpowiedzieć na pytanie.
#     Odpowiedz tylko na podstawie kontekstu.
#     Jeśli odpowiedź nie znajduje się w kontekście, odpowiedz "Nie wiem".

#     Pytanie:
#     {query}

#     Odpowiedź:
#     """
#     return prompt.strip()

# def process_text_and_store(text, chunk_size, chunk_overlap, prompts):
#     # 1. Split text into chunks
#     splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
#     chunks = splitter.split_text(text)

#     # 2. Create embeddings
#     embedding_model = OllamaEmbeddings(model="llama3.2:latest")  # Replace with your local model name
#     embeddings = embedding_model.embed_documents(chunks)

#     # 3. Calculate cosine similarity matrix
#     similarity_matrix = cosine_similarity(embeddings)

#     # 4. Initialize agent
#     llm = Ollama(model="hf.co/mradermacher/Llama-PLLuM-70B-instruct-GGUF:Q5_K_M", temperature=0.9, max_tokens=1024)  # Replace with your local model name
#    # Run Ollama model on full text
#     #llm = Ollama(model="llama3")
#     full_text_info = llm.invoke(f"Extract key insights from the following HTML:\n{html_text}")

#     # Run agent on each chunk
#     tools = [Tool(name="OllamaExtractor", func=llm.invoke, description="Extract info from chunk")]
#     agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION)

    
#     #chain = prompt | model

#     response = chain.invoke({"question": "Jakie są rodzaje podatków w Polsce?"})


#     # 5. Extract information using prompts
#     extracted_data = []
#     for i, chunk in enumerate(chunks):
#         chunk_info = {"chunk_index": i, "chunk_text": chunk, "similarities": similarity_matrix[i].tolist()}
#         for prompt in prompts:
#             response = chain.invoke((f"{prompt}\n\n{chunk}")
#             chunk_info[prompt] = response
#         extracted_data.append(chunk_info)


def find_similar_chunks(
    query: str,
    db_config: dict,
    table_name: str = "text_chunks",
    top_k: int = 5
):
    # 1. Create embedding for query
    embedder = OllamaEmbeddings(model="llama2")
    query_embedding = np.array(embedder.embed_query(query))

    # 2. Fetch all embeddings from DB
    with connect(**db_config, row_factory=dict_row) as conn:
        with conn.cursor() as cur:
            cur.execute(f"SELECT chunk, embedding FROM {table_name};")
            rows = cur.fetchall()

    # 3. Compute cosine similarity
    def cosine_similarity(a, b):
        a, b = np.array(a), np.array(b)
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    similarities = [
        (row["chunk"], cosine_similarity(query_embedding, row["embedding"]))
        for row in rows
    ]

    # 4. Return top_k most similar chunks
    similarities.sort(key=lambda x: x[1], reverse=True)
    return similarities[:top_k]



