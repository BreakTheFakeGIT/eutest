import sys
sys.path.append("/dane/eutest")
import os
import time
from multiprocessing import Pool, cpu_count

from bs4 import BeautifulSoup
from langchain.text_splitter import RecursiveCharacterTextSplitter

#from src.extract.extract_eu_json import single_json_line
from extract_eu_json_test import single_json_line


import src.utils.logger as logger_utils

from dotenv import load_dotenv
load_dotenv()

logger = logger_utils.setup_logger(name='multiprocessing_eu_json')


POSTGRES_DSN = os.environ.get("POSTGRES_DSN_TEST")
CONSOLIDATE_JSON_FOLDER = os.environ.get("CONSOLIDATE_JSON_FOLDER")


# Function to process the entire file using multiprocessing
def process_json_file(file_path):
    """ Process the entire JSON file using multiprocessing """
    logger.info(f'Start processing file: {file_path}')
    with open(file_path, 'r', encoding="utf8", errors="ignore") as f:
        lines = f.readlines()
    logger.info(f'Number of lines to process: {len(lines)}')
    # Use Pool to process lines in parallel
    with Pool(processes=24) as pool:
        results = pool.map(single_json_line, lines)
    pool.close()
    #pool.terminate()
    return results

dir_list = os.listdir(CONSOLIDATE_JSON_FOLDER)
# Process the file and get the result
for file in dir_list:
    file_path = os.path.join(CONSOLIDATE_JSON_FOLDER, file)
     # Skip files that do not match the pattern
    if not file.startswith('consolidate_json_') or not file.endswith('.json'):
        logger.info(f'Skipping file: {file}')
        continue

    t = time.time()
    results = process_json_file(file_path)
    # Print first 1 results for verification
    for result in results[0:1]:
        #logger.info(f"Results process_json_file for verification JSON object: {result}")
        logger.info(f'Tax: {result.get('typ_podatku', [])}')
        logger.info(f'Interp_HTML: {result.get('tresc_interesariusz_html', [])}')



        html_text = result.get('tresc_interesariusz_html', [])
        # Parse HTML
        soup = BeautifulSoup(html_text, "html.parser")
        # Define tags to include and exclusion rules
        include_tags = ['h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'p', 'b', 'strong']
        exclude_classes = ['exclude']
        exclude_phrases = ['Dodatkowe', 'Pouczenie', 'Uzasadnienie','Ocena stanowiska']

        # Extract relevant text
        included_texts = []
        for tag in soup.find_all(include_tags):
            # Exclude by class
            if any(cls in exclude_classes for cls in tag.get('class', [])):
                continue

            text = tag.get_text(strip=True)

            # Exclude by phrase
            if any(phrase.lower() in text.lower() for phrase in exclude_phrases):
                break

            included_texts.append(text)

        # Combine extracted text
        combined_text = "\n".join(included_texts)

        # Split into chunks using LangChain
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=100
        )
        chunks = text_splitter.split_text(combined_text)

        # Output chunks
        for i, chunk in enumerate(chunks):
            print(f"Chunk {i+1}:\n{chunk}\n")
        print(chunks)


    # Save to postgres
    # from etl_process.src.postgres import save_to_postgres
    # save_to_postgres(POSTGRES_DSN, results)



    t = time.time() - t
    logger.info(f'End processing file. Time: {t:.2f} seconds. Number of processed JSON objects: {len(results)}. CPU count: {cpu_count()}. File: {file}')
    del results



