import pandas as pd
import re
import os
from os.path import join
import src.utils.logger as logger_utils
from dotenv import load_dotenv
load_dotenv()

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', None)

###################################
EU_DICT_URL = os.environ.get("EU_DICT_URL")
EU_DICT = os.environ.get("EU_DICT")

###################################
logger = logger_utils.setup_logger(name="eu_load_dict")
logger.info(f'EU_DICT_URL: {EU_DICT_URL}')
logger.info(f'EU_DICT: {EU_DICT}')

logger_utils.create_directories(EU_DICT)

###################################
KEYWORD_GROUPS = {
    'vat|Podatek od towarów i usług (VAT)': ['[vat]', '[jpk]', '39610'],
    'pit|Podatek dochodowy od osób fizycznych (PIT)': ['[pit]', '[vzm]', '[zpdof]'],
    'cit|Podatek dochodowy od osób prawnych (CIT)': ['[cit]', '[wzp]', '[wzs]'],
    'pcc|Podatek od czynności cywilnoprawnych (PCC)': ['[pcc]'],
    'psd|Podatek od spadków i darowizn': ['[psd]'],
    'akcyza|Podatek akcyzowy': ['[akcyza]', 'ustawa z dnia 23 stycznia 2004 r. o podatku akcyzowym'],
    'op|Ordynacja podatkowa': ['[op]', '57875', 'e-deklaracje', 'ustawa o doręczeniach elektronicznych'],
    'gry|Ustawa o grach hazardowych': ['[gry]', '39593'],
    'malpki|Ustawa o wychowaniu w trzeźwości i przeciwdziałaniu alkoholizmowi': ['[małpki]'],
    'spw|Ustawa o specjalnym podatku węglowodorowym': ['[spw]'],
    'pt|Podatek tonażowy': ['[pt]'],
    'pkop|Ustawa o podatku od wydobycia niektórych kopalin': ['[p-kop]'],
    'spdet|Ustawa podatek od sprzedaży detalicznej': ['[sprzedaż-detal]'],
    'fin|Ustawa o podatku od niektórych instytucji finansowych': ['[fin]'],
    'cukier|Ustawa o zdrowiu publicznym': ['[cukier]'],
    'wip|Ustawa o wymianie informacji podatkowych z innymi państwami': ['[wip]'],
    'globe|Ustawa o opodatkowaniu wyrównawczym jednostek składowych grup międzynarodowych i krajowych': ['[globe]'],
    'nip|Ustawa o zasadach ewidencji i identyfikacji podatników i płatników': ['[nip]'],
}


def remove_files(path: str, prefix):
    """ Remove files in the specified path that match the given prefix """
    import glob
    pattern = os.path.join(path, f"{prefix}_*_*.*")
    files = glob.glob(pattern)
    logger.info(f"Found {len(files)} files to delete.")
    for file in files:
        try:
            os.remove(file)
            logger.info(f"Deleted: {file}")
        except Exception as e:
            import traceback
            logger.error(f"Error deleting {file}: {e}")
            logger.error(traceback.print_tb(e.__traceback__))

def assign_multiple_tax(text):
    """ Function """
    result = []
    series = pd.Series(text)
    for key, keywords in KEYWORD_GROUPS.items():
        # Escape special characters in keywords
        escaped_keywords = [re.escape(k) for k in keywords]
        pattern = '|'.join(escaped_keywords)
        if series.str.contains(pattern).any():
            result.append(key)
    return '_'.join(result) if result else 'inne'

def split_and_rename_column(data: pd.DataFrame, column_name:str, new_col1:str, new_col2:str, sep='|'):
    """ Function """
    new_cols = data[column_name].str.split(sep, expand=True)
    data[new_col1] = new_cols[0]
    data[new_col2] = new_cols[1]
    return data

def data_eu_dict(data: pd.DataFrame):
    """ Function """
    data = data.drop_duplicates()
    data.columns = [col.lower() for col in data.columns]
    data['dataczas_akt'] = logger_utils.set_datetime_local()
    data['id'] = data['id'].astype(int)
    data['slownikid'] = data['slownikid'].astype(int)
    data['kod'] = data['kod'].astype(str).str.lower()
    data['opis'] = data['opis'].astype(str).str.lower()
    data['wartosc'] = data['wartosc'].astype(str).str.lower()
    data.reset_index(drop=True, inplace=True, names=None)
    data.sort_values(by=['id'], inplace=True, ignore_index=True, ascending=False)

    # Filter for relevant slownikid values
    # 5 - słownik słow kluczowych
    # 18 - słownik zagadnień 
    # 19 - słownik przepisów
    # data = data[data['slownikid'].isin([5,18,19])] 

    data['typ_podatku_list'] = data['opis'].apply(lambda x: re.findall(r'\[[^\[\]]+\]', x))
    data['typ_podatku_tech'] = data['id'].astype(str) + '' + data['typ_podatku_list'].astype(str)
    data['typ_podatku_tech2'] = data['typ_podatku_tech'].apply(lambda x: assign_multiple_tax([x]))
    split_and_rename_column(data=data, column_name='typ_podatku_tech2', new_col1='typ_podatku', new_col2='typ_podatku_opis')
    for col in ['typ_podatku_list','typ_podatku','typ_podatku_opis']:
        data.loc[(data['slownikid'] != 19), 'typ_podatku_list'] = None
    data = data.drop(columns=['typ_podatku_tech','typ_podatku_tech2'])
    return data

def get_page_to_data(url: str) -> pd.DataFrame:

    import requests
    from io import StringIO
    try:
        response = requests.get(url)
        logger.info(f"Fetched data from {url} with status code {response.status_code}")
    except Exception as e:
        import traceback
        logger.error(f"Error fetching data from {url}: {e}")
        logger.error(traceback.print_tb(e.__traceback__))

    if response.status_code == 200:
        logger.info(f"Response content (first 500 chars): {response.text[:500]}")
        data = pd.read_json(StringIO(response.text))
        data = data_eu_dict(data=data)
    else:
        logger.error(f"Failed to fetch data, status code, data frame is empty: {response.status_code}")
        logger.error(f"Response content (first 500 chars): {response.text[:500]}")
        data = pd.DataFrame()
    return data

def main_process():

    logger.info(f'KEYWORD_GROUPS {KEYWORD_GROUPS}')

    # Create base path with timestamp
    prefix = "eu_dictionary"
    base_path = logger_utils.base_path_with_timestamp(path=EU_DICT, prefix=prefix)
    logger.info(f'Base path: {base_path}')

    # Clean up old files
    remove_files(path=EU_DICT, prefix=prefix)

    # Fetch data 
    data = get_page_to_data(EU_DICT_URL)

    # Display data information
    logger.info(f'Data shape: {data.shape}')
    logger.info(f'Data columns: {data.columns.tolist()}')
    logger.info(f'Data head:\n{data.head()}')

    # Save to CSV and pickle
    data.to_csv(f"{base_path}.csv", encoding='utf-8', index=False, header=True, sep=';')
    data.to_excel(f"{base_path}.xlsx", index=False, header=True, sheet_name='slownik_eureka')
    data.to_pickle(f"{base_path}.pkl")

    # Read back the saved files to verify
    read_csv = pd.read_csv(f"{base_path}.csv", sep=';')
    logger.info(f'Read CSV shape: {read_csv.shape}')
    logger.info(f'Read CSV head:\n{read_csv.head()}')
    read_excel = pd.read_excel(f"{base_path}.xlsx", sheet_name='slownik_eureka')
    logger.info(f'Read Excel shape: {read_excel.shape}')
    logger.info(f'Read Excel head:\n{read_excel.head()}')
    read_pkl = pd.read_pickle(f"{base_path}.pkl")
    logger.info(f'Read pickle shape: {read_pkl.shape}')
    logger.info(f'Read pickle head:\n{read_pkl.head()}')

    del data, read_csv, read_excel, read_pkl


if __name__ == "__main__":
    main_process()













