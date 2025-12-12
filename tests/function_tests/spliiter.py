import re
from typing import List, Dict, Optional, Type, Any

from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    CharacterTextSplitter,
    TextSplitter,
)
import psycopg

import os
from dotenv import load_dotenv
load_dotenv()
POSTGRES_DSN = os.environ.get("POSTGRES_DSN")

def experiment_text_splitting(
    text: str,
    chunk_sizes: List[int],
    chunk_overlaps: List[int],
    splitter_class: Type[TextSplitter] = RecursiveCharacterTextSplitter,
    exclude_phrase: Optional[str] = None,
    exclude_word: Optional[str] = None,
    exclude_regex: Optional[str] = None,
    **splitter_kwargs,
) -> Dict[str, List[str]]:
    """
    Splits text into chunks using different parameters and allows filtering.
    """
    results = {}

    # Pre-compile regex patterns for efficiency if they are provided
    regex_pattern = re.compile(exclude_regex) if exclude_regex else None
    # Use word boundaries (\b) to ensure we match whole words only (case-insensitive)
    word_pattern = (
        re.compile(r"\b" + re.escape(exclude_word) + r"\b", re.IGNORECASE)
        if exclude_word
        else None
    )

    print(f"--- Starting Splitting Experiments on {splitter_class.__name__} ---")
    for size in chunk_sizes:
        for overlap in chunk_overlaps:
            # Basic validation: overlap should be less than chunk size
            if overlap >= size:
                print(f"Skipping experiment: overlap ({overlap}) >= size ({size})")
                continue

            experiment_key = f"size_{size}_overlap_{overlap}"
            print(f"\n=> Running experiment: {experiment_key}")

            # 1. Initialize the specified text splitter
            try:
                text_splitter = splitter_class(
                    chunk_size=size, chunk_overlap=overlap, **splitter_kwargs
                )
            except Exception as e:
                print(f"Error initializing splitter for {experiment_key}: {e}")
                continue

            # 2. Split the text into chunks
            all_chunks = text_splitter.split_text(text)

            # 3. Filter the chunks based on exclusion criteria
            filtered_chunks = []
            for chunk in all_chunks:
                # Assume the chunk is valid until proven otherwise
                is_excluded = False

                # Check for phrase exclusion (case-sensitive)
                if exclude_phrase and exclude_phrase in chunk:
                    is_excluded = True
                    print(f"   - Excluding chunk (contains phrase '{exclude_phrase}')")

                # Check for whole word exclusion (case-insensitive)
                if not is_excluded and word_pattern and word_pattern.search(chunk):
                    is_excluded = True
                    print(f"   - Excluding chunk (contains word '{exclude_word}')")

                # Check for regex exclusion
                if not is_excluded and regex_pattern and regex_pattern.search(chunk):
                    is_excluded = True
                    print(f"   - Excluding chunk (matches regex '{exclude_regex}')")

                if not is_excluded:
                    filtered_chunks.append(chunk)

            results[experiment_key] = filtered_chunks
            print(
                f"   Original chunks created: {len(all_chunks)}, "
                f"Chunks after filtering: {len(filtered_chunks)}"
            )

    return results


def write_results_to_postgres(
    results: Dict[str, List[str]], db_params: str
) -> None:
    """
    Writes the results of text splitting experiments to a PostgreSQL database.
    """
    create_table_sql = """
    CREATE TABLE IF NOT EXISTS text_splitting_results (
        id SERIAL PRIMARY KEY,
        experiment_key VARCHAR(255) NOT NULL,
        chunk_index INTEGER NOT NULL,
        chunk_content TEXT NOT NULL,
        created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
    );
    """
    insert_sql = """
    INSERT INTO text_splitting_results (experiment_key, chunk_index, chunk_content)
    VALUES (%s, %s, %s);
    """

    conn = None
    try:
        # Connect to the PostgreSQL database
        print("\n--- Connecting to PostgreSQL database... ---")
        conn = psycopg.connect(db_params)
        print("Connection successful.")

        with conn.cursor() as cur:
            # Create the table if it doesn't exist
            print("Ensuring 'text_splitting_results' table exists...")
            cur.execute(create_table_sql)
            print("Table check complete.")

            # Iterate through the results and insert into the database
            total_chunks_inserted = 0
            for experiment_key, chunks in results.items():
                print(f"   - Inserting data for experiment: {experiment_key}")
                for i, chunk in enumerate(chunks):
                    cur.execute(insert_sql, (experiment_key, i, chunk))
                total_chunks_inserted += len(chunks)

            # Commit the transaction
            conn.commit()
            print(f"\nSuccessfully inserted {total_chunks_inserted} chunks into the database.")

    except (Exception, psycopg.Error) as error:
        print(f"Error while connecting to or writing to PostgreSQL: {error}")
        if conn:
            # Rollback the transaction in case of error
            conn.rollback()
    finally:
        # Closing database connection.
        if conn:
            conn.close()
            print("PostgreSQL connection is closed.")


def retrieve_chunks_from_postgres(
    db_params: Dict[str, str], experiment_key: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Selects and retrieves chunked text data from the PostgreSQL database.
    """
    base_select_sql = """
    SELECT id, experiment_key, chunk_index, chunk_content, created_at
    FROM text_splitting_results
    """
    params = []

    if experiment_key:
        base_select_sql += " WHERE experiment_key = %s ORDER BY chunk_index;"
        params.append(experiment_key)
    else:
        base_select_sql += " ORDER BY experiment_key, chunk_index;"

    results = []
    conn = None
    try:
        # Connect to the PostgreSQL database
        print("\n--- Connecting to PostgreSQL database for retrieval... ---")
        conn = psycopg.connect(**db_params)
        print("Connection successful.")

        with conn.cursor() as cur:
            # Execute the query
            print(f"Executing query to retrieve data for key: '{experiment_key or 'ALL'}'")
            cur.execute(base_select_sql, params)
            
            # Fetch all rows
            rows = cur.fetchall()
            
            if not rows:
                print("No records found for the given criteria.")
                return []

            # Get column names from the cursor description
            colnames = [desc[0] for desc in cur.description]
            for row in rows:
                results.append(dict(zip(colnames, row)))
            
            print(f"Successfully retrieved {len(results)} records.")

    except (Exception, psycopg.Error) as error:
        print(f"Error while connecting to or reading from PostgreSQL: {error}")
    finally:
        # Closing database connection.
        if conn:
            conn.close()
            print("PostgreSQL connection is closed.")
    
    return results


# This block demonstrates how to use the function.
if __name__ == "__main__":
    sample_text = """
    LangChain is a framework for developing applications powered by language models.
    It is designed to be data-aware and agentic. This means it can connect a language
    model to other sources of data, and allow a language model to interact with its
    environment. The core idea of the library is that we can “chain” together different
    components to create more advanced use cases around LLMs. Chains may consist of
    multiple components from several modules. A key part of this is the '[DO NOT PROCESS]' component.

    These components can include model wrappers, prompt templates, and indexes. It's a
    versatile tool for any developer. Another '[DO NOT PROCESS]' keyword is here for testing
    exclusion. LangChain provides a standard interface for chains, lots of integrations
    with other tools, and end-to-end chains for common applications. This is a test
    sentence with the word 'developer' that we might want to exclude. This is test 123.
    """

    # --- Experiment Parameters ---
    chunk_sizes_to_test = [150, 250]
    chunk_overlaps_to_test = [30, 60]

    # --- 1. Basic Experiment: No exclusions ---
    print("\n" + "=" * 50)
    print("### EXPERIMENT 1: No Exclusions ###")
    print("=" * 50)
    results_no_exclusion = experiment_text_splitting(
        text=sample_text,
        chunk_sizes=chunk_sizes_to_test,
        chunk_overlaps=chunk_overlaps_to_test,
    )
    # Print one of the results for inspection
    if results_no_exclusion:
        first_key = list(results_no_exclusion.keys())[0]
        print(f"\nSample result for '{first_key}':")
        for i, chunk in enumerate(results_no_exclusion[first_key][:2]): # Show first 2 chunks
             print(f"  Chunk {i+1}: '{chunk[:100]}...'")


    # --- 2. Exclusion by Phrase ---
    print("\n" + "=" * 50)
    print("### EXPERIMENT 2: Exclude by phrase '[DO NOT PROCESS]' ###")
    print("=" * 50)
    results_phrase_exclusion = experiment_text_splitting(
        text=sample_text,
        chunk_sizes=chunk_sizes_to_test,
        chunk_overlaps=chunk_overlaps_to_test,
        exclude_phrase="[DO NOT PROCESS]",
    )

    # --- 3. Exclusion by Word ---
    print("\n" + "=" * 50)
    print("### EXPERIMENT 3: Exclude by whole word 'developer' ###")
    print("=" * 50)
    results_word_exclusion = experiment_text_splitting(
        text=sample_text,
        chunk_sizes=chunk_sizes_to_test,
        chunk_overlaps=chunk_overlaps_to_test,
        exclude_word="developer",
    )

    # --- 4. Exclusion by Regex ---
    print("\n" + "=" * 50)
    print("### EXPERIMENT 4: Exclude by regex for numbers (\\d+) ###")
    print("=" * 50)
    results_regex_exclusion = experiment_text_splitting(
        text=sample_text,
        chunk_sizes=chunk_sizes_to_test,
        chunk_overlaps=chunk_overlaps_to_test,
        exclude_regex=r"\d+",
    )

    # --- 5. Using a different splitter ---
    print("\n" + "=" * 50)
    print("### EXPERIMENT 5: Using CharacterTextSplitter ###")
    print("=" * 50)
    # This splitter is simpler and splits only on a specified character.
    results_different_splitter = experiment_text_splitting(
        text=sample_text,
        chunk_sizes=chunk_sizes_to_test,
        chunk_overlaps=chunk_overlaps_to_test,
        splitter_class=CharacterTextSplitter,
        separator="\n", # Kwarg passed to CharacterTextSplitter
    )

    # --- 6. Write results to PostgreSQL Database ---
    print("\n" + "=" * 50)
    print("### EXPERIMENT 6: Writing to PostgreSQL ###")
    print("=" * 50)



    # Example of writing the results from the first experiment
    # You can change this to any other result set you want to save.
    print("Attempting to write 'results_no_exclusion' to the database.")
    print("NOTE: This will fail if you don't update connection details and have a running PostgreSQL server.")
    write_results_to_postgres(results_no_exclusion, db_connection_params)

    # --- 7. Retrieve Chunked Data from PostgreSQL ---
    print("\n" + "=" * 50)
    print("### EXPERIMENT 7: Retrieving Data from PostgreSQL ###")
    print("=" * 50)

    # Example of retrieving the data we just wrote for a specific experiment
    retrieved_data = retrieve_chunks_from_postgres(
        db_params=POSTGRES_DSN,
        experiment_key="size_150_overlap_30" # Should match a key from 'results_no_exclusion'
    )

    if retrieved_data:
        print(f"\nSuccessfully retrieved {len(retrieved_data)} chunks for 'size_150_overlap_30'.")
        print("First retrieved chunk:")
        # Print the first result pretty
        first_chunk = retrieved_data[0]
        for key, value in first_chunk.items():
            if key == 'chunk_content':
                 print(f"  {key}: '{str(value)[:100]}...'")
            else:
                 print(f"  {key}: {value}")

