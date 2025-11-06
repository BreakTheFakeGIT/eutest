
import sys
sys.path.append("/dane/eutest")
import os
from os.path import join
import pandas as pd
import re
import src.utils.logger as logger_utils
from dotenv import load_dotenv
load_dotenv()

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', None)

logger = logger_utils.setup_logger(name="dictionary_test")

path = EU_DICT = os.environ.get("EU_DICT")
logger.info(f"EU_DICT path: {EU_DICT}")

path_file = [file for file in os.listdir(path) if re.match(r'eu_dictionary_.*\.csv', file)][0]
path_file = os.path.join(path, path_file)
logger.info(f"Reading data from {path}")

data = pd.read_csv(path_file,
    sep=';',
    engine='python',
    encoding='utf-8',
    usecols=['id','opis','wartosc','slownikid'],
    dtype={'id': int, 'opis': str,'wartosc': str, 'slownikid': int}
)

#print(data.head())

KEYWORD_GROUPS = {
    'vat': ['[vat]', '[jpk]', '39610'],
    'pit': ['[pit]', '[vzm]', '[zpdof]'],
    'cit': ['[cit]', '[wzp]', '[wzs]'],
    'pcc': ['[pcc]'],
    'psd': ['[psd]'],
    'akcyza': ['[akcyza]', 'ustawa z dnia 23 stycznia 2004 r. o podatku akcyzowym'],
    'op': ['[op]', '57875', 'e-deklaracje', 'ustawa o doręczeniach elektronicznych'],
    'gry': ['[gry]', '39593'],
    'malpki': ['[małpki]'],
    'spw': ['[spw]'],
    'pt': ['[pt]'],
    'pkop': ['[p-kop]'],  # Uncomment if needed
    'spdet': ['[sprzedaż-detal]'],
    'fin': ['[fin]'],
    'cukier': ['[cukier]'],
    'wip': ['[wip]'],
    'globe': ['[globe]'],
}

def assign_multiple_tax(text):
    result = []
    series = pd.Series(text)
    for key, keywords in KEYWORD_GROUPS.items():
        # Escape special characters in keywords
        escaped_keywords = [re.escape(k) for k in keywords]
        pattern = '|'.join(escaped_keywords)
        if series.str.contains(pattern).any():
            result.append(key)
    return '_'.join(result) if result else 'inne'


# Function to create new columns for each unique sequence
def extract_bracket_sequences(data, column_name):
    """"""
    # Step 1: Extract all bracketed sequences from each row
    bracket_sequences = data[column_name].apply(lambda x: re.findall(r'\[[^\[\]]+\]', x))

    # Step 2: Flatten and get unique sequences
    unique_sequences = sorted(set(seq for sublist in bracket_sequences for seq in sublist))

    # Step 3: Create new columns for each unique sequence
    for seq in unique_sequences:
        data[seq] = bracket_sequences.apply(lambda x: seq in x)

    return data



def data_dict(data: pd.DataFrame):
    data = data.drop_duplicates()
    data.columns = [col.lower() for col in data.columns]
    data['dataczas_akt'] = logger_utils.set_datetime_local()
    data['id'] = data['id'].astype(int)
    data['slownikid'] = data['slownikid'].astype(int)
    #data['kod'] = data['kod'].astype(str).str.lower()
    data['opis'] = data['opis'].astype(str).str.lower()
    data['wartosc'] = data['wartosc'].astype(str).str.lower()
    data.reset_index(drop=True, inplace=True, names=None)
    data.sort_values(by=['id'], inplace=True, ignore_index=True, ascending=False)

   #Filter for relevant slownikid values
    # 5 - słownik słow kluczowych
    # 18 - słownik zagadnień 
    # 19 - słownik przepisów
    data = data[data['slownikid'].isin([5,18,19])] 
    data['typ_podatku_list'] = data['opis'].apply(lambda x: re.findall(r'\[[^\[\]]+\]', x))
    data['typ_podatku_tech'] = data['id'].astype(str) + '' + data['typ_podatku_list'].astype(str)
    data['typ_podatku'] = data['typ_podatku_tech'].apply(lambda x: assign_multiple_tax([x]))
    data.loc[(data['slownikid'] != 19), 'typ_podatku'] = None
    data = data.drop(columns=['typ_podatku_tech'])
    return data


data = data_dict(data=data)
data = data[['opis','typ_podatku_list','typ_podatku']]
print(data.head())



# df = pd.DataFrame({
#     'text_column': [
#         'Informacja o [vat] i [pit] w systemie e-deklaracje',
#         'Nowa ustawa o podatku akcyzowym',
#         'Raport finansowy i sprzedaż detaliczna',
#         'Brak powiązania z żadnym podatkiem'
#     ]
# })

# df['category'] = df['text_column'].apply(lambda x: compare_and_concatenate([x]))
# print(df)







# # Define a function using regex
# def assign_info(text):
#     if pd.Series(text).str.contains(r'cit', case=False, regex=True).any():
#         return 'cit-abc'
#     elif pd.Series(text).str.contains(r'vat', case=False, regex=True).any():
#         return 'vat-xyz'
#     elif pd.Series(text).str.contains(r'pit', case=False, regex=True).any():
#         return 'pit-nml'
#     else:
#         return 'unknown'

# Apply the function to each row
# df['new_info'] = df['text_column'].apply(assign_info)

# print(df)

# Sample DataFrame
# df = pd.DataFrame({
#     'text_column': [
#         'this is about cit tax',
#         'vat applies here',
#         'pit is relevant',
#         'no match here',
#         'cit and vat both apply',
#         'pit and cit are mentioned',
#         'all: cit, vat, pit'
#     ]
# })

# # Function to detect and combine multiple matches
# def assign_multiple_info(text):
#     result = []
#     if re.search(r'cit', text, re.IGNORECASE):
#         result.append('cit-abc')
#     if re.search(r'vat', text, re.IGNORECASE):
#         result.append('vat-xyz')
#     if re.search(r'pit', text, re.IGNORECASE):
#         result.append('pit-nml')
#     return ' + '.join(result) if result else 'unknown'

# # Apply the function
# df['new_info'] = df['text_column'].apply(assign_multiple_info)

# print(df)






# # Function to create new columns for each unique sequence
# def extract_bracket_sequences(df, column_name):
#     """"""
#     # Step 1: Extract all bracketed sequences from each row
#     bracket_sequences = df[column_name].apply(lambda x: re.findall(r'\[[^\[\]]+\]', x))

#     # Step 2: Flatten and get unique sequences
#     unique_sequences = sorted(set(seq for sublist in bracket_sequences for seq in sublist))

#     # Step 3: Create new columns for each unique sequence
#     for seq in unique_sequences:
#         df[seq] = bracket_sequences.apply(lambda x: seq in x)

#     return df


# def extract_tax():
#     data = data['opis'].apply(lambda x: re.findall(r'\[[^\[\]]+\]', x))






























