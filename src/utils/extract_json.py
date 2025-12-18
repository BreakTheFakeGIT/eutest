import re
import os
import json
import math
import pandas as pd
from datetime import datetime
import pickle
from src.utils.text_process import TextEuJson
from src.utils.dbsql import read_sql_files_for_table
import src.utils.logger as logger_utils
from dotenv import load_dotenv
load_dotenv()

###################################
TABLE_NAME = os.environ.get("TBL_INTERP")
EU_DICT = os.environ.get("EU_DICT")
EU_SQL = os.environ.get("EU_SQL")
READ_SQL = read_sql_files_for_table(table_name=TABLE_NAME, input_dir=EU_SQL)
INSERT_SQL = READ_SQL.get('insert_sql',[])
COLUMNS = READ_SQL.get('columns_sql',[])

###################################
logger = logger_utils.setup_logger(name=f'src_utils_extract_json_{TABLE_NAME}')
logger.info(f"TABLE_NAME path: {TABLE_NAME}")
logger.info(f"EU_DICT path: {EU_DICT}")
logger.info(f"EU_SQL path: {EU_SQL}")
logger.info(f"COLUMNS from EU_SQL: {COLUMNS}")

###################################

def get_eu_dict(path: str) -> tuple[dict, dict]:
    """ Read eu_dict CSV file and return two dictionaries """
    path_file = [file for file in os.listdir(path) if re.match(r'eu_dictionary_.*\.csv', file)][0]
    path_file = os.path.join(path, path_file)
    data = pd.read_csv(path_file,
        sep=';',
        engine='python',
        encoding='utf-8',
        usecols=['id','opis','wartosc','slownikid','typ_podatku','typ_podatku_opis'],
        dtype={'id': int, 'opis': str,'wartosc': str, 'slownikid': int, 'typ_podatku': str,'typ_podatku_opis': str}
        )
    # Filter for relevant slownikid values
    # 5 - słownik słow kluczowych
    # 18 - słownik zagadnień
    # 19 - słownik przepisów
    data = data[data['slownikid'].isin([5,18,19])] 
    #logger.info(f"Data shape: {data.shape} | Columns: {data.columns.tolist()}")
    #logger.info(f"Data sample:\n{data.head(3).to_dict(orient='records')}")
    eu_dict_opis = data.set_index('id')['opis'].to_dict()
    #logger.info(f"Data eu_dict_opis sample: {list(eu_dict_opis.items())[:5]}")
    eu_dict_wartosc = data.set_index('id')['wartosc'].to_dict()
    #logger.info(f"Data eu_dict_wartosc sample: {list(eu_dict_wartosc.items())[:5]}")
    eu_dict_typ_podatku = data[data['slownikid'].isin([19])].set_index('id')['typ_podatku'].to_dict()
    #logger.info(f"Data eu_dict_typ_podatku sample: {list(eu_dict_typ_podatku.items())[:5]}")
    eu_dict_typ_podatku_opis = data[data['slownikid'].isin([19])].set_index('id')['typ_podatku_opis'].to_dict()
    #logger.info(f"Data eu_dict_typ_podatku_opis sample: {list(eu_dict_typ_podatku_opis.items())[:5]}")

    return eu_dict_opis, eu_dict_wartosc, eu_dict_typ_podatku, eu_dict_typ_podatku_opis

def join_list_and_dict(string_list, value_dict):
    """ Join list of strings with dictionary values """
    joined_values = []
    for number_str in string_list:
        try:
            key = int(number_str)
            if key in value_dict:
                value = value_dict[key]
                joined_values.append(value)
                #logger.info(f"Found a match for key {key}: '{value}'")
            # else:
            #     logger.info(f"Key {key} from the list was not found in the dictionary.")
        except ValueError:
            logger.info(f"Could not process '{number_str}' as it is not a valid number.")
    return joined_values

def clean_and_join_list(input_list):
    """
    Cleans a list by removing None, NaN, and duplicates,
    sorts it, and joins it into a comma-separated string.
    """
    try:
        input_list = list(set(input_list))
        cleaned = {
            item for item in input_list
            if item is not None and not (isinstance(item, float) and math.isnan(item))
        }
        sorted_list = sorted(map(str,input_list))

        return '_'.join(sorted_list)
    except Exception as e:
        return logger.info(f"An error occurred: {e}")

def single_json_line(json_line):
    """ Process a single JSON object """
    data = json.loads(json_line)
    dict1_keys = ['versionId', 'nazwa', 'szablonId', 'wersjaSzablonuId']
    dict1 = {k.lower(): v for k, v in data.items() if k in dict1_keys}

    document = data.get('dokument').get('fields')
    dict2_keys =  ['ID_INFORMACJI','SLOWA_KLUCZOWE','PRZEPISY','ZAGADNIENIA','KATEGORIA_INFORMACJI',
                'STATUS_INFORMACJI','DATA_PUBLIKACJI','TEZA','DT_WYD','SYG','TRESC_INTERESARIUSZ']
    dict2 = [key for key in document if key.get('key') in dict2_keys]
    dict2 = {key.get('key').lower(): key.get('value') for key in dict2 if key.get('key') in dict2_keys}

    if 'tresc_interesariusz' in dict2:
        dict2['tresc_interesariusz_html'] = dict2['tresc_interesariusz']
        text_processor = TextEuJson(text=dict2['tresc_interesariusz'])
        try:
            text = text_processor.process().to_string()
            dict2['tresc_interesariusz'] = text
        except KeyError:
            logger.info(f"Key 'tresc_interesariusz' not found in the dictionary. {dict2.items()}")
    else:
        dict2['tresc_interesariusz'] = None
        dict2['tresc_interesariusz_html'] = None
        logger.info(f"Key 'tresc_interesariusz' not found in the dictionary. {dict2.items()}")

    eu_dict_opis, eu_dict_wartosc, eu_dict_typ_podatku, eu_dict_typ_podatku_opis = get_eu_dict(path=EU_DICT)

    list_id_eu = dict2.get('przepisy', []) + dict2.get('zagadnienia', []) + dict2.get('slowa_kluczowe', [])
    list_id_eu = [int(item) for item in list_id_eu]
    list_kod_eu = ['przepis']*len(dict2.get('przepisy', [])) + ['zagadnienie']*len(dict2.get('zagadnienia', [])) + ['slowo_kluczowe']*len(dict2.get('slowa_kluczowe', []))
    dict2['id_eu'] = list_id_eu
    dict2['kod_eu'] = list_kod_eu

    if list_id_eu:
        dict2['opis_eu'] = join_list_and_dict(string_list=list_id_eu, value_dict=eu_dict_opis)
        dict2['wartosc_eu'] = join_list_and_dict(string_list=list_id_eu, value_dict=eu_dict_wartosc)
        typ_podatku = join_list_and_dict(string_list=list_id_eu, value_dict=eu_dict_typ_podatku)
        dict2['typ_podatku'] = clean_and_join_list(typ_podatku)
        typ_podatku_opis = join_list_and_dict(string_list=list_id_eu, value_dict=eu_dict_typ_podatku_opis)
        dict2['typ_podatku_opis'] = clean_and_join_list(typ_podatku_opis)
    else:
        for key in [
            'id_eu', 'kod_eu', 'opis_eu', 'wartosc_eu',
            'typ_podatku', 'typ_podatku_opis', 'przepisy',
            'zagadnienia', 'slowa_kluczowe'
        ]:
            dict2[key] = None

    keys_to_check = ['przepisy', 'zagadnienia', 'slowa_kluczowe']
    for key in keys_to_check:
        values = dict2.get(key, [])
        if values:
            dict2[f'{key}_opis_eu'] = join_list_and_dict(string_list=values, value_dict=eu_dict_opis)
            dict2[f'{key}_wartosc_eu'] = join_list_and_dict(string_list=values, value_dict=eu_dict_wartosc)
        else:
            dict2[f'{key}_opis_eu'] = None
            dict2[f'{key}_wartosc_eu'] = None
            logger.info(f"Key: {key.upper()} not found in the dictionary. ID_INFORMACJI: {dict2.get('id_informacji',0)}.")

    dict2['dataczas_akt'] = datetime.now().strftime("%Y%m%d %H%M%S")
    #merged_dict = {**dict1, **dict2}
    merged_dict = dict2 | dict1
    #merged_dict = {key: merged_dict[key] for key in COLUMNS if key in merged_dict}

    return merged_dict