import os
import json
import src.utils.logger as logger_utils
from dotenv import load_dotenv
load_dotenv()

###################################
JSON_FOLDER = os.environ.get("JSON_FOLDER")
CONSOLIDATE_JSON_FOLDER = os.environ.get("CONSOLIDATE_JSON_FOLDER")

###################################
logger = logger_utils.setup_logger(name="eu_consolidate_json")
logger.info(f'JSON_FOLDER: {JSON_FOLDER}')

logger_utils.create_directories(CONSOLIDATE_JSON_FOLDER)
logger.info(f'CONSOLIDATE_JSON_FOLDER: {CONSOLIDATE_JSON_FOLDER}')

###################################

def consolidate_json_files(input_folder: str, output_folder: str, files_per_group:int, limit_files=0):
    json_files = [f for f in os.listdir(input_folder) if f.endswith('.json')]

    json_files.sort(reverse=True, key=lambda x: int(x.split('_')[-1].split('.')[0]))
    logger.info(f"Sorted JSON files (total {len(json_files)} files): {json_files[:5]}.")
    if limit_files>0:
        json_files = json_files[:limit_files]
        logger.info(f"Limit to first {len(json_files)} files processing): {json_files}.")

    consolidated_data = []
    file_count = 0

    for i, file_name in enumerate(json_files):
        with open(os.path.join(input_folder, file_name), 'r', encoding="utf8", errors="ignore") as file:
            for line in file:
                try:
                    json_data = json.loads(line)
                    consolidated_data.append(json_data)
                except json.JSONDecodeError as e:
                    import traceback
                    logger.error(f"JSONDecodeError: {e.msg} at line {e.lineno} column {e.colno} file {file}")
                    logger.error(traceback.print_tb(e.__traceback__))

        if (i + 1) % files_per_group == 0:
            output_file_name = f'consolidate_json_{file_count + 1}.json'
            with open(os.path.join(output_folder, output_file_name), 'w',encoding="utf8", errors="ignore") as output_file:
                for item in consolidated_data:
                    output_file.write(json.dumps(item, ensure_ascii=False) + '\n')

            consolidated_data = []
            file_count += 1

    if consolidated_data:
        output_file_name = f'consolidate_json_{file_count + 1}.json'
        with open(os.path.join(output_folder, output_file_name), 'w', encoding="utf8", errors="ignore") as output_file:
            for item in consolidated_data:
                output_file.write(json.dumps(item, ensure_ascii=False) + '\n')


def main_process(files_per_group, limit_files):
    remove_old_files = [f for f in os.listdir(CONSOLIDATE_JSON_FOLDER) if f.startswith('consolidate_json_') and f.endswith('.json')]
    remove_old_files.sort(reverse=True, key=lambda x: int(x.split('_')[-1].split('.')[0]))
    for f in remove_old_files:
        os.remove(os.path.join(CONSOLIDATE_JSON_FOLDER, f))
    logger.info(f"Removed old consolidated files: {remove_old_files}")

    consolidate_json_files(input_folder=JSON_FOLDER, output_folder=CONSOLIDATE_JSON_FOLDER, files_per_group=files_per_group, limit_files=limit_files)
    logger.info("Consolidation complete.")

    file_consolidated = os.listdir(CONSOLIDATE_JSON_FOLDER)
    logger.info(f'Consolidated files: {file_consolidated}')
    for file_name in file_consolidated:
        logger.info(f'Consolidated file: {file_name}')
        read_consolidated_file = os.path.join(CONSOLIDATE_JSON_FOLDER, file_name)
        with open(read_consolidated_file, 'r', encoding="utf8", errors="ignore") as file:
            count_lines = sum(1 for line in file)
            logger.info(f"Total lines in {read_consolidated_file}: {count_lines}")
            for i, line in enumerate(file):
                if i < 5:  # Print first 5 lines
                    logger.info(f'{line.strip()}')
                else:
                    break
        logger.info(f"Displayed first 5 lines of {read_consolidated_file}.")
        logger.info("Displayed first 5 lines of the consolidated file.")
        logger.info("Consolidation complete.")


if __name__ == "__main__":

    main_process(files_per_group=20, limit_files=0)

