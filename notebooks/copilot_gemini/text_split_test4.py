import psycopg
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from typing import Dict, Any, Type

def split_and_store_text(
    data: Dict[int, Dict[str, Any]],
    splitter_class: Type,
    chunk_size: int,
    chunk_overlap: int,
    db_params: Dict[str, str]
) -> None:
    """
    Splits text from a dictionary using a specified Langchain splitter and stores
    the chunks in a PostgreSQL database.
    """
    # 1. Initialize the specified text splitter from Langchain
    try:
        text_splitter = splitter_class(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            is_separator_regex=False,
        )
        print(f"Using '{splitter_class.__name__}' for splitting.")
    except Exception as e:
        print(f"Error initializing splitter: {e}")
        return

    # 2. Establish connection to the PostgreSQL database
    conn = None
    try:
        print("Connecting to the PostgreSQL database...")
        conn = psycopg.connect(**db_params)
        cur = conn.cursor()

        # 3. Create the table for storing chunks if it doesn't already exist
        # This ensures the code can be run multiple times without error.
        cur.execute("""
            CREATE TABLE IF NOT EXISTS text_chunks (
                chunk_id SERIAL PRIMARY KEY,
                original_id INTEGER NOT NULL,
                tax VARCHAR(255),
                chunk_text TEXT NOT NULL,
                created_at TIMESTAMPTZ DEFAULT NOW()
            );
        """)
        print("Table 'text_chunks' is ready.")

        # 4. Iterate through the data, split text, and insert into the database
        total_chunks = 0
        for original_id, content in data.items():
            tax = content.get('tax', '')
            text_to_split = content.get('text', '')

            if not text_to_split:
                print(f"Skipping ID {original_id} due to empty text.")
                continue

            # Use the Langchain splitter to create the chunks
            chunks = text_splitter.split_text(text_to_split)
            
            print(f"ID {original_id}: Splitting text into {len(chunks)} chunks.")

            # Insert each chunk as a separate row
            for chunk in chunks:
                cur.execute(
                    "INSERT INTO text_chunks (original_id, tax, chunk_text) VALUES (%s, %s, %s)",
                    (original_id, tax, chunk)
                )
            total_chunks += len(chunks)

        # 5. Commit all the transactions to save the changes
        conn.commit()
        print(f"\nSuccessfully inserted a total of {total_chunks} chunks into the database.")

    except (psycopg.OperationalError, psycopg.DatabaseError) as error:
        print(f"Database error: {error}")
        if conn:
            # Rollback changes if any error occurs during the transaction
            conn.rollback()
    except Exception as error:
        print(f"An unexpected error occurred: {error}")
        if conn:
            conn.rollback()
    finally:
        # 6. Always close the cursor and connection
        if conn:
            if 'cur' in locals() and cur:
                cur.close()
            conn.close()
            print("Database connection closed.")

# --- Example Usage ---
if __name__ == '__main__':
    # 1. Sample data structure
    sample_data = {
        101: {
            'tax': 'document_type_a',
            'text': (
                "LangChain is a framework for developing applications powered by language models. "
                "It enables applications that are: data-aware, agentic, and differentiated. "
                "Data-aware applications can connect a language model to other sources of data. "
                "Agentic applications allow a language model to interact with its environment. "
                "This framework is essential for building complex AI systems."
            )
        },
        102: {
            'tax': 'report_section_b',
            'text': (
                "Psycopg 3 is the latest major version of the most popular PostgreSQL database adapter for the Python programming language. "
                "It is a complete rewrite of the original Psycopg 2, offering better performance, asynchronous support, "
                "and a more modern feature set. It uses the 'libpq' C library for a fast and direct connection to the database server."
            )
        },
         103: {
            'tax': 'user_feedback_c',
            'text': (
                "Text splitting is a critical step in preparing data for language models. "
                "The goal is to break down large documents into smaller, semantically meaningful chunks. "
                "This is important because models have a limited context window. The overlap parameter helps "
                "maintain context between chunks, preventing loss of information at the boundaries."
            )
        }
    }

    # 2. Database connection parameters
    # IMPORTANT: Replace these with your actual PostgreSQL credentials.
    db_connection_params = {
        "dbname": "your_db_name",
        "user": "your_username",
        "password": "your_password",
        "host": "localhost",  # or your db host
        "port": "5432"
    }

    # 3. Call the function with your desired settings
    # You can easily swap 'RecursiveCharacterTextSplitter' with another splitter like 'CharacterTextSplitter'
    split_and_store_text(
        data=sample_data,
        splitter_class=RecursiveCharacterTextSplitter,
        chunk_size=120,
        chunk_overlap=20,
        db_params=db_connection_params
    )