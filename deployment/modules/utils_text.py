"""" Text Function """
import pytz
import datetime
import re
import string
import pandas as pd
from collections import Counter
from collections import OrderedDict
import spacy
from spacy.lang.pl.examples import sentences
import jellyfish

nlp = spacy.load("pl_core_news_sm")
print(nlp._path)

def nlp_doc(text: str):
    """ Function """
    return nlp(text)

def nlp_get_list_text_pos(text: str, tags: list[str]) -> str:
    """ Function """
    doc = nlp_doc(text)
    new_text =''
    for token in doc:
        if(token.is_stop or token.is_punct or token.is_currency or token.is_space or len(token.text) <= 3 or token.pos_ == "NUM"): #Remove Stopwords, Punctuations, Currency and Spaces
            pass
        elif token.pos_ in tags: #['ADJ', 'NOUN']:
            new_text = " ".join((new_text, token.text.lower()))
    return new_text

def nlp_get_list_lemma_pos(text: str, tags: list[str]) -> str:
    """ Function """
    doc = nlp_doc(text)
    new_text =''
    for token in doc:
        if(token.is_stop or token.is_punct or token.is_currency or token.is_space or len(token.text) <= 3 or token.pos_ == "NUM"): #Remove Stopwords, Punctuations, Currency and Spaces
            pass
        elif token.pos_ in tags: #['ADJ', 'NOUN']:
            new_text = " ".join((new_text, token.lemma_.lower()))
    return new_text

def split_on_window_startwith(sequence: str, word:str, limit:int):
    """ Function """
    results = []
    split_sequence = sequence.split()
    iteration_length = len(split_sequence) - (limit - 1)
    max_window_indicies = range(iteration_length)
    for index in max_window_indicies:
        join_sequence = ' '.join(split_sequence[index:index + limit])
        if join_sequence.startswith(word):
            results.append(join_sequence)

    return results

def nlp_extract_word_startwith(text: str, word: str, limit: int):
    ls_word = split_on_window_startwith(sequence=text,word=word,limit=limit)
    ls_word_pos = []
    for w in ls_word:
        ls_word = nlp_get_list_text_pos(text=w, tags = ['ADJ','NOUN'])
        ls_word_pos.append(ls_word)
    return ls_word_pos

def split_on_window_contains(sequence: str,  word:str, limit:int):
    """ Function """
    results = []
    split_sequence = sequence.split()
    iteration_length = len(split_sequence) - (limit - 1)
    max_window_indicies = range(iteration_length)
    for index in max_window_indicies:
        join_sequence = ' '.join(split_sequence[index:index + limit])
        if join_sequence.__contains__(word):
            results.append(join_sequence)

    return results

def nlp_common_words(text: str, n_cw: int):
    """ Function """
    doc = nlp_doc(text)
    words = [token.text
            for token in doc
            if not token.is_stop and not token.is_punct]

    word_freq = Counter(words)
    common_words = word_freq.most_common(n_cw)
    return common_words

def nlp_common_words_tag(text: str, n_cw: int, tag: str):
    """ Function """
    doc = nlp_doc(text)
    words = [token.text
            for token in doc
            if (not token.is_stop and
                not token.is_punct and
                token.pos_ == tag)]

    word_freq = Counter(words)
    common_words = word_freq.most_common(n_cw)
    return common_words

def ngrams(text:str, n:int):
    """ Function """
    text = text.split(' ')
    output = []
    for i in range(len(text)-n+1):
        output.append(text[i:i+n])

    return [' '.join(x) for x in output]

def nlp_extract_entities(text: str):
    """ Function """
    doc = nlp_doc(text)
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    return entities

def sim_two_words(text: str):
    """ Function """
    words = text.split()
    return jellyfish.jaro_similarity(words[0], words[1])

def contains_number(text:str):
    return re.search('\d', text)

def list_order_text(text: str) -> str:
    """ Function """
    words = text.split()
    words = list(OrderedDict.fromkeys(words))
    words = " ".join(words)
    return words.strip()

def unique_list(list_text: list) -> list:
    """ Function """
    list_text = list(set(list_text))
    return list_text

def create_seq_list(r1, r2):
    """ Function """
    return list(range(r1, r2+1))

def create_data_from_list(list_text: list, name: str) -> pd.DataFrame:
    """ Function """
    data = pd.DataFrame(list(zip([name]*len(list_text), list_text)),
                        columns = ['obszar','opis'])

    mask = data['opis'].astype(str).str.len() > 3
    data.loc[mask, 'opis'] = data.loc[mask, 'opis']
    return data

def proces_doc_vector(ls_doc_tresc: str):
    """ Function """
    list_out_words = ['fakt','stan','mowa','dzień','dnia','dni','dacie','opis','wskaz','rozum','samy',
                    'wnios','zakre','ustaw','interp','indywid','zadan','stoso','związ','postan','możliw',
                    'skorzyst','zawart','niepraw','prawi','okreś','danym','miesi','orzecz','organ','wyrok',
                    'myśl','brzmie','względ','zwany','stron','powyż', 'państ', 'pozost', 'podsta',
                    'lit','ust','art','rama','jedne','cel','skarb','ocen','zamian','poję','zastrz','zasad',
                    'przypad','wymien','uzasad','naczel','przepis','prawa','ważne','ważne','będąc',
                    'dyrektyw','uwag','dokon','załącz','świetl','regulac','zapisów','grudn','maj','listopad',
                    'marca','stycz','lip','racji','momencie','innym','dyrektor','tymi','pierwsz','pyta',
                    'znaczenia','decyduj','razem','okres','dane','marca','momen','dotycz','piśmie','pismo',
                    'czas','wedlug','charakter','kąt','sygnat','źródł','istot','inne','roczne','dalsz',
                    'chwil','obecn','luty','grona','czerw','uwadz','data','wpływ','same','sierpn','jaki',
                    'wstęp','kwesti','dany','dana','lata','inną','wyraz','defini','szczegól','jedną','datę',
                    'brak','pogląd','zbieżn','przyszł','znacząc','letni','punkcie','rozporządz','początek',
                    'kolei','punkt','widzeni','odrębną','krajowej','informacji','skarbowej','pan','polsk',
                    'minist','finansów','rzeczy','szanow','stwierdza','wydanie','skutków', 'podatkowych',
                    'sprawie', 'zdarze','znak','amach','ordyna','podatkowa','dalec','koniecz','wąski','zdani',
                    'tamże','różn','chęć','niemożliwe','cytow','rawidłow','daty','treś'
                    ]
    ls_doc_ner = nlp_extract_entities(text=ls_doc_tresc)
    if ls_doc_ner:
        for record in ls_doc_ner:
            ner_value, ner_type = record
            if ner_type == 'date' or ner_type == 'persName' or ner_type == 'time' or ner_type == 'geogName':
                ls_doc_tresc = ls_doc_tresc.replace(ner_value, "")

    ls_doc_word_art = split_on_window_startwith(sequence=ls_doc_tresc, word='art', limit=4)
    ls_doc_word_art = unique_list(ls_doc_word_art)

    for word in list_out_words:
        ls_doc_tresc = ' '.join(x for x in ls_doc_tresc.split() if not x.startswith(word))

    ls_doc_split = nlp_get_list_text_pos(text=ls_doc_tresc, tags = ['ADJ','VERB','NOUN'])
    ls_doc_split = split_text_by_words(text=ls_doc_split,n_words=10)
    ls_doc_split = unique_list([list_order_text(x) for x in ls_doc_split if len(x.split())==10 and not contains_number(x)])

    ls_doc_word_czy = split_on_window_startwith(sequence=ls_doc_tresc, word='czy ', limit=10)
    ls_doc_word_nie = split_on_window_startwith(sequence=ls_doc_tresc,word='nie', limit=10)

    ls_doc_word = []
    for word in ['ulg','gmin','osoba','nieruchom','lokal','spół','jednoosob','darowi','spadek',
                    'stawk','podat','opodat','placów','wynagro','renta','emeryt','ryczałt','koszt','przychód','dochód']:
        doc_word_start = split_on_window_startwith(sequence=ls_doc_tresc, word=word, limit=20)
        doc_word_start = ' '.join(doc_word_start)
        doc_word_start = nlp_get_list_text_pos(text=doc_word_start, tags = ['ADJ','VERB','NOUN'])
        doc_word_start = split_on_window_startwith(sequence=doc_word_start, word=word, limit=10)
        #print(doc_word_start)
        if doc_word_start:
            doc_word_start = unique_list([list_order_text(x) for x in doc_word_start if len(x.split())>1 and not contains_number(x)])
            #print(word)
            #print(doc_word_start)
            ls_doc_word+=doc_word_start
    #print('tutaj ls_doc_word')
    #print(ls_doc_word)

    ls_doc = nlp_get_list_text_pos(text=ls_doc_tresc, tags = ['ADJ','NOUN','NUM'])
    ls_doc = ngrams(ls_doc,n=2)
    ls_doc = unique_list(ls_doc)
    ls_doc = [x for x in ls_doc if len(x.split())==2 and sim_two_words(x)<0.8]
    ls_doc = sorted(ls_doc, key=len, reverse=True)
    #ls_doc = [x for x in ls_doc if len(x.split())==2 and not contains_number(x) and sim_two_words(x)<0.8]

    text_fields = [
            ('doc_bigram',ls_doc),
            ('doc_art',ls_doc_word_art),
            ('doc_czy',ls_doc_word_czy),
            ('doc_podzal',ls_doc_split),
            ('doc_nie',ls_doc_word_nie),
            ('doc_slowa',ls_doc_word)
            ]


    data_emb = []
    for record in text_fields:
        match record:
            case [obszar,text]:
                data = create_data_from_list(list_text=text, name=obszar)
                data_emb.append(data)

        # PROCES EMBEDDINGS
    data_embedding = pd.concat(data_emb,ignore_index=True)
    data_embedding = data_embedding.groupby('obszar')['opis'].apply(list).reset_index(name='opis')
    ls_text = data_embedding['opis'].tolist()
    ls_field = data_embedding['obszar'].tolist()
    #ls_model_tpod_lemma = nlp_get_list_lemma_pos(text=ls_doc_tresc, tags = ['ADJ','NOUN'])
    return ls_field, ls_text

def to_string(object):
    """ Funkcja """
    return str(object)

def lower_case_string(text: str):
    """ Funkcja """
    return text.lower()

def sub_html(text: str, regex: str):
    """ Funkcja """
    return regex.sub(' ', text)

def remove_html(text: str):
    """ Funkcja """
    for char in (r'<[^>]+>', r'\\n', r'&nbsp;',r'\\t'):
        regex = re.compile(char)
        text = sub_html(text, regex)
    return text

def convert_char(text: str) -> str:
    """ Function """
    text = text.replace(u'.', u' ')
    text = text.replace(u';', u' ')
    text = text.replace(u'"', u' ')
    text = text.replace(u' sp. ', u' spółka ')
    text = text.replace(u' sp ', u' spółka ')
    text = text.replace(u' s.a. ', u' spółka akcyjna ')
    text = text.replace(u'pkob', u'polska klasyfikacja obiektów budowlanych')
    text = text.replace(u'pkd', u'polska klasyfikacja działalności')
    text = text.replace(u'pkwiu', u'polska klasyfikacja wyrobów i usług')
    text = text.replace(u' cn ', u' nomenklatura scalona ')
    return text


def remove_special_char(text: str) -> str:
    """ Function """
    for char in "-\n()[]:„”·-–/…•;":
        text = text.replace(char, " ")
    return text

def remove_special_phrase(text: str) -> str:
    """ Function """
    for char in ['tzn','tzw.',' cyt.',' cyt ',' itd.',' ad.',' zm.',' pn.',
                 ' pn ',' późn.', ' późn ',' zw.',' ul.',' ad ',' ok. ',
                 ' np. ',' np ', ' sygn. ',' sygn ',' m. in.',' m.in.','ww.',
                 ' nr. ',' nr ', 'tj.',' t. j. ',' t.j.','dz urz.','dz. urz.',
                 ' dz u. ',' dz. u. ','„(…)”','\n']:
        text = text.replace(char, " ")
    return text

def text_trim(text: str):
    """ Funkcja """
    return text.strip()

def remove_newline(text: str):
    """ Funkcja """
    return ' '.join(text.splitlines())

def remove_bs4_xa0(text: str):
    """ Funkcja """
    return text.replace(u'\xa0', u'')

def cut_text_end_word(text: str, pattern: str) -> str:
    """ Function """
    try:
        match=(re.search(pattern, text))
        idx = match.start()
    except:
        idx = len(text)
    return text[0:idx]

def remove_seq_dot(text: str) -> str:
    """ Function """
    return re.sub(r'\s(\.){2,}\s', ' ', text)

def remove_all_whitespaces(text: str):
    """ Funkcja """
    return re.sub(r'\s+', ' ', text, flags=re.UNICODE)

def simple_clean_text(text: str) -> str:
    """ Function """
    text = to_string(text)
    text = lower_case_string(text)
    text = remove_html(text)
    text = convert_char(text)
    text = remove_special_phrase(text)
    text = remove_special_char(text)
    text = text_trim(text)
    text = remove_bs4_xa0(text)
    text = remove_newline(text)
    for pat in ['stronie przysługuje prawo do','skarg','poucze','informacja o zakresie'
                ,'informacja zakresie','postępowanie przed sądami administ','zażalenie na postan','dodatkowe informacje']:
        text = cut_text_end_word(text, pattern=pat)
    text = remove_seq_dot(text)
    text = remove_all_whitespaces(text)
    return text

def get_text_list_st_history(session) -> list[str]:
    """ Funkcja """
    text_list = [key.get('content')[0] for key  in session if key.get('role') == 'user']
    return [str(item.get('text')) for item in text_list]

def get_text_from_text_list(text_list: list[str]) -> str:
    """ Funkcja """
    text_list = list(dict.fromkeys(text_list))
    return ' '.join(text_list)

def first_half_text(text: str) -> str:
    return text[0:(len(text)//2)+1]

def proces_text(text_input: str, text_session: list[str]):
    text_list = [text_input]
    text_list = get_text_from_text_list(text_list=text_list)
    text_model = simple_clean_text(text=text_list)
    ls_field, ls_text = proces_doc_vector(text_model)
    if text_session:
        text_list_se = get_text_list_st_history(session=text_session)
        text_list_se = get_text_from_text_list(text_list=text_list_se)
        text_model_se = simple_clean_text(text=text_list_se)
        ls_field_se, ls_text_se = proces_doc_vector(text_model_se)
        text_list = text_list + text_list_se
        ls_field = ls_field + ls_field_se
        ls_text =  ls_text + ls_text_se

    content_type, content = criteria_text_input(text_input=text_input, text_list=text_list)

    # ls_text_half = nlp_get_list_text_pos(text=text_model, tags = ['NOUN'])
    # ls_text_half = unique_list([first_half_text(x) for x in ls_text_half.split()])
    # ls_text_half = [x for x in ls_text_half if len(x)>4 or not contains_number(x)]


    #ls_field, ls_text, ls_model_tpod_lemma = proces_doc_vector(text_model)
    #return type, content, ls_field, ls_text, ls_model_tpod_lemma, ls_text_half
    return content_type, content, ls_field, ls_text

# KRYTERIA
def numbers_of_words(text: str) -> int:
    """ Funkcja """
    return len(text.split())

def criteria_text_input(text_input: str, text_list: str):
    """ Funkcja """
    cnt_text_input = numbers_of_words(text_input)
    #cnt_text_list = ut.numbers_of_words(text_list)
    text_list = text_list.split()
    len_text_list = len(text_list)
    cut_off_len_text_list_param = 6400
    cut_off_cnt_text_input = 10

    if len_text_list<=cut_off_len_text_list_param and cnt_text_input>cut_off_cnt_text_input:
        type = 'data_frame'
        content = '**Wyszukane interpertacje**'

    elif len_text_list>cut_off_len_text_list_param:
        type = 'long_text'
        content = f'**Brak wyszukania - łączny tekst jest zadługi. Liczba znaków z Promptów wynosi {len_text_list} i przekracza wartość graniczną {cut_off_len_text_list_param} znaków. Oceń oraz Wyczyść historię**'

    elif cnt_text_input<=cut_off_cnt_text_input:
        type = 'short_text'
        content = f'**Brak wyszukania -  wprowadzony tekst jest za krótki. Składa się z {cnt_text_input} słów/słowa, poniżej wartości granicznej {cut_off_cnt_text_input} słów.***'

    else:
        type = 'other_text'
        content = f'**Inne kryteria: liczba znaków - {len_text_list}, liczba słów - {cnt_text_input}. Oceń oraz Wyczyść historię***'

    return type, content

def set_datetime_local():
    """ Funkcja """
    #set the timezone
    tz_info = pytz.timezone('Europe/Warsaw')
    dt_format = f'%Y-%m-%d %H:%M:%S %Z'
    return datetime.datetime.now(tz=tz_info).strftime(dt_format)

def remove_special_chars(text: str) -> str:
    """ Funkcja """
    for char in "%=,–·,„”‒!@#$^&*_+|\":;'<>./-…€-\\n()[]:„”·-–/…•":
        text = text.replace(char, " ")
    return text

def remove_text_right_bracket(text):
    """ Funkcja """
    return re.sub("[0-9a-z]+\)"," ",text)

def remove_punctation(text: str) -> str:
    """ Funkcja """
    _punctuation = set(string.punctuation)
    for punct in set(text).intersection(_punctuation):
        text= text.replace(punct, ' ')
    return ' '.join(text.split())

def remove_digits(text: str) -> str:
    """ Funkcja """
    return re.sub(r'[0-9]+', ' ', text)

def remove_non_alnum_translate(text: str):
    """ Funkcja """
    table = str.maketrans('', '', string.punctuation)
    return text.translate(table).replace(' ', ' ')

def split_text_by_words(text: str, n_words: int):
    """ Funkcja """
    words = text.split()
    subs = []
    for i in range(0, len(words), n_words):
        subs.append(" ".join(words[i:i+n_words]))
    return subs

def extract_text_st_history(session):
    """ Funkcja """
    text_list = [key.get('content')[0] for key  in session if key.get('role') == 'user']
    text_list = [str(item.get('text')) for item in text_list]
    text_list = list(dict.fromkeys(text_list))
    return ' '.join(text_list)

def get_df_list_st_history(session) -> list:
    """ Funkcja """
    text_list = [key.get('content')[0] for key  in session if key.get('role') == 'assistant']
    #text_list = [item.get('data') for item in text_list]
    return text_list

def clean_text(object):
    """ Funkcja """
    text = to_string(object)
    text = lower_case_string(text)
    text = remove_html(text)
    text = remove_newline(text)
    text = remove_bs4_xa0(text)
    #text = replace_dot_for_original_sentence(text)
    text = convert_char(text)
    #text = remove_special_chars(text)
    for pat in ['stronie przysługuje prawo do','skarg','poucze','informacja o zakresie'
                ,'informacja zakresie','postępowanie przed sądami administ','zażalenie na postan','dodatkowe informacje']:
        text = cut_text_end_word(text, pattern=pat)
    text = remove_seq_dot(text)
    text = text_trim(text)
    text = remove_all_whitespaces(text)
    return text



def proces_text_from_st_history(session_st_history: list) -> list:
    """ Funkcja """
    text = extract_text_st_history(session_st_history)
    text = clean_text(text)
    text = split_text_by_words(text,200)
    return text

def create_url(id_informacji):
    """ Funkcja """
    return f'https://eureka.mf.gov.pl/informacje/podglad/{id_informacji}'

def make_clickable_both(text: str):
    name, url = text.split('#')
    return f'<a target="_blank" href="{url}">{name}</a>'




# INNE
def find_sentences(word_list, sentence_list):
    result = []
    for sentence in sentence_list:
        # For each sentence, check if all the words are in the sentence
        if any([re.search(w, sentence) for w in word_list]):
            result.append(sentence)
    return result



if __name__ == '__main__':
    pass