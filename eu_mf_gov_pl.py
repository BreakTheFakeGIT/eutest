import traceback
import asyncio
import aiohttp
import json
import aiofiles
import os
import src.utils.logger as logger_utils
from src.utils.logger import set_datetime_local, diff_lists_set
from src.utils.select import select_id_informacji
from dotenv import load_dotenv
load_dotenv()

logger = logger_utils.setup_logger(name="eu_mf_gov_pl")

###################################
TBL_INTERP = os.environ.get("TBL_INTERP")
POSTGRES_DSN = os.environ.get("POSTGRES_DSN")

###################################

def range_id_infromacji(logger, TBL_INTERP, POSTGRES_DSN, START_ID: int = -1, END_ID: int =-1):
    #664945
    if START_ID == -1 or END_ID == -1:
        try: 
            rows = select_id_informacji(conn_str=POSTGRES_DSN, table_name=TBL_INTERP)
            logger.info(f'ROWS: select_id_informacji:\nLEN: {len(rows)}, START {min(rows)} END {max(rows)}\nLIST: {rows[0:10]}.')
            START_ID = min(rows)
            END_ID = max(rows)
            RANGE_ID = list(range(START_ID, END_ID, 1))
            logger.info(f'RANGE_ID BEFORE diff_lists_set:\nLEN: {len(RANGE_ID)}, START {START_ID} END {END_ID}\nLIST: {RANGE_ID[0:10]}.')
            RANGE_ID, _ = diff_lists_set(list1=RANGE_ID, list2=rows)
            logger.info(f'RANGE_ID AFTER diff_lists_set:\nLEN: {len(RANGE_ID)}, START {min(RANGE_ID)} END {max(RANGE_ID)}\nLIST: {RANGE_ID[0:10]}.')

        except Exception as e:
            logger.error(f'Error in select_id_informacji: {e}')
            logger.error(traceback.format_exc())

    else:
        RANGE_ID = list(range(START_ID, END_ID, 1))
        logger.info(f'The range of IDs from {START_ID} to {END_ID}. RANGE_ID: {len(RANGE_ID)}')

    RANGE_ID.sort(reverse=True)
    logger.info(f'Sorted RANGE_ID: {RANGE_ID[0:10]}')
    return RANGE_ID



# --- Configuration ---
# The range of IDs to fetch, from START_ID to END_ID.
RANGE_ID = range_id_infromacji(logger, TBL_INTERP, POSTGRES_DSN)

# The base URL for the API endpoint.
BASE_URL = 'https://eureka.mf.gov.pl/api/public/v1/informacje'

# The number of JSON records to store in a single output file.
RECORDS_PER_FILE = 1000

# The delay in seconds between each download request.
REQUEST_DELAY_SECONDS = 0.1

# The directory where the output files will be saved.
#OUTPUT_DIR = "/dane/eutest/data/complete_json/"
OUTPUT_DIR = os.environ.get("COMPLETE_JSON_FOLDER")

# The template for the output filenames.
FILENAME_TEMPLATE = "consolidate_json_{}_{}.json"


async def fetch_data(session, url, retries=3, delay=5):
    """
    Asynchronously fetches JSON data from a given URL with retries on failure.

    Args:
        session (aiohttp.ClientSession): The client session for making requests.
        url (str): The URL to fetch data from.
        retries (int): Number of times to retry the request if it fails.
        delay (int): Delay in seconds between retries.

    Returns:
        dict: The fetched JSON data as a dictionary, or None if fetching fails after all retries.
    """
    for attempt in range(retries):
        try:
            async with session.get(url) as response:
                # Check for a successful response (HTTP 200 OK)
                if response.status == 200:
                    return await response.json()
                else:
                    logger.info(f"Error fetching {url}: Received status code {response.status}")
                    # No need to retry for common client/server errors like 404 or 500
                    return None
        except aiohttp.ClientError as e:
            logger.error(f"Request failed for {url}: {e}. Attempt {attempt + 1} of {retries}.")
            if attempt < retries - 1:
                await asyncio.sleep(delay)
            else:
                logger.error(f"All retries failed for {url}.")
                return None

async def save_to_file(data_list, file_path):
    """
    Asynchronously saves a list of JSON objects to a file, with each object on a new line.

    Args:
        data_list (list): A list of JSON serializable objects (dictionaries).
        file_path (str): The path to the file where the data will be saved.
    """
    logger.info(f"Saving {len(data_list)} records to {file_path}...")
    try:
        # Use aiofiles for non-blocking file I/O
        async with aiofiles.open(file_path, mode='w', encoding='utf-8') as f:
            for item in data_list:
                # Convert the Python dictionary to a JSON string and write it as a new line.
                # ensure_ascii=False is important for correct character encoding.
                await f.write(json.dumps(item, ensure_ascii=False) + '\n')
        logger.info(f"Successfully saved data to {file_path}")
    except IOError as e:
        logger.error(f"Error writing to file {file_path}: {e}")


async def main():
    """
    The main asynchronous function to coordinate the process of fetching and saving data.
    """
    # Create the output directory if it doesn't already exist.
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        logger.info(f"Created output directory: {OUTPUT_DIR}")

    # Use a single aiohttp.ClientSession for connection pooling and better performance.
    async with aiohttp.ClientSession() as session:
        collected_data = []
        file_counter = 1

        # Loop through all the IDs in the specified range, in descending order.
        for item_id in RANGE_ID:
            url = f'{BASE_URL}/{item_id}'
            logger.info(f"Requesting data for ID: {item_id}...")

            data = await fetch_data(session, url)

            if data:
                collected_data.append(data)

                # When the collected data reaches the batch size, save it to a file.
                if len(collected_data) >= RECORDS_PER_FILE:
                    file_path = os.path.join(OUTPUT_DIR, FILENAME_TEMPLATE.format(item_id,set_datetime_local()))
                    await save_to_file(collected_data, file_path)

                    # Reset the list for the next batch and increment the file counter.
                    collected_data = []
                    file_counter += 1

            # Wait for the specified delay before making the next request.
            logger.info(f"Waiting for {REQUEST_DELAY_SECONDS} seconds...")
            await asyncio.sleep(REQUEST_DELAY_SECONDS)

        # After the loop finishes, save any remaining data that hasn't been written yet.
        if collected_data:
            file_path = os.path.join(OUTPUT_DIR, FILENAME_TEMPLATE.format(item_id,set_datetime_local()))
            await save_to_file(collected_data, file_path)

    logger.info("\nDownload process has finished.")

if __name__ == "__main__":
    #This block runs the main asynchronous function.
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.error("\nProcess interrupted by user.")
