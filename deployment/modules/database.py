"""" chat eureka"""
import itertools
import streamlit as st
import pandas as pd
import psycopg2
#from pgvector.psycopg2 import register_vector # !!!!!!!!
from psycopg2 import sql
import psycopg2.extras as extras
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from itertools import combinations
import pickle 
from functools import wraps
from time import time

def timeit(f):
    @wraps(f)
    def wrap(*args, **kw):
        ts = time()
        result = f(*args, **kw)
        te = time()-ts
        # print('Exectution time function:%r args:[%r, %r] took: %2.4f sec' % \
        #   (f.__name__, args, kw, te-ts))
        print('Exectution time function:%r took: %2.4f sec' % (f.__name__, te))
        return result , format(te, '2.4f')
    return wrap

pd.options.mode.copy_on_write = False
pd.options.mode.chained_assignment = None

@st.cache_resource
def init_connection():
    """ Funkcja """
    #dialect = st.secrets.db_credentials.dialect
    host = st.secrets.db_credentials.host
    port = st.secrets.db_credentials.port
    database = st.secrets.db_credentials.database
    username = st.secrets.db_credentials.username
    password = st.secrets.db_credentials.password
    return psycopg2.connect(dbname=database, user=username, host=host, port=port, password=password)

@st.cache_resource
def set_model_type(path_model:str):
    return SentenceTransformer(path_model, device='cpu')

@st.cache_data
def pkl_load_slownik(file_input:str):
    data = pd.read_pickle(file_input)
    return data

def generate_combinations(word_list):
    all_combinations = []
    for r in range(1, len(word_list) + 1):
        comb = combinations(word_list, r)
        all_combinations.extend(comb)
    return sorted(all_combinations, key=lambda x: (len(x), x), reverse=True)

def generate_sql_query(combinations_list):
    case_clauses = []
    for combo in combinations_list:
        combo_name = "|".join(combo)
        condition = " AND ".join([f"opis LIKE '%{word}%'" for word in combo])
        case_clause = f" WHEN {condition} THEN '{combo_name.replace(' ','_')}' "
        case_clauses.append(case_clause)
    query = "SELECT \n id_informacji, CASE " + " \n".join(case_clauses) + " END AS komb_tags \nFROM t0"
    return query

def generate_product(lst1: list, lst2: list):
    return list(itertools.product(lst1, lst2))

def add_opis_like_to_sql(text: str):
    text = """ opis LIKE '%"""+text.lower()+"""%' """
    return text

def calculate_query_list_words(keywords:str):
    """ Function """
    try:
        if keywords:
            if len(keywords)==1:
                query_key = f' '+add_opis_like_to_sql(text=keywords[0])
            else:
                query_key = f' OR '.join([add_opis_like_to_sql(text=item) for item in keywords])
        return f' ('+query_key+') '

    except Exception as e:
        print('Exception - calaculate query:', e)
        return " opis LIKE '%podatek%' "

def get_embedding_for_user_query(model: object, text:str):
    """ Funkcja """
    user_query_embedding = model.encode([text], normalize_embeddings=True)
    return user_query_embedding[0].tolist()

def get_zero_iter_from_db(conn,
                          keywords: list[str],
                          query_to_db: str,
                          sql_keywords:str,
                          query_keywords:str,
                          schema: str, table: str):
    """ Function """
    try:
        schema_table = f'{schema}.{table}'
        #sql = """SELECT id_informacji, nazwa_pliku FROM """ + str(schema_table) + """ ORDER BY id_informacji DESC;"""
        if len(keywords)==1:
            sql_where_tags = """ 1=1 """
        elif len(keywords)==2:
            sql_where_tags = """ 1=1 AND ( ile_tagow>1 OR ( drank_cnt=1 AND ile_tagow=1 ) )"""
        else:
            sql_where_tags = """ 1=1 AND ( ile_tagow>1 OR ( drank_cnt<"""+str(len(keywords)-1)+""" AND ile_tagow=1 ) )"""

        sql = """ WITH t0 AS (
            SELECT id_informacji, opis FROM
                (
                    SELECT
                        id_informacji
                        , string_agg( DISTINCT opis , ' ' ORDER BY opis) AS opis
                    FROM """ + str(schema_table) + """
                    WHERE 1=1
                        AND obszar = 'sl_wartosc'
                        AND """ + str(query_to_db) + """
                    GROUP BY
                        id_informacji
                ) AS t
            WHERE 1=1
                AND """ + str(query_keywords) + """
            ),
            t1 AS (
             """ + str(sql_keywords) + """
            ),
            t2 AS (
            SELECT 
                id_informacji
                , komb_tags
                , (LENGTH(komb_tags) - LENGTH(replace(komb_tags, '|', '')))+1 AS ile_tagow
                ,COUNT(id_informacji) OVER(PARTITION BY komb_tags) AS cnt
            FROM t1
            ),
            t3 AS (
            SELECT * FROM (
                SELECT 
                    id_informacji
                    , komb_tags
                    , ile_tagow
                    , cnt
                    , DENSE_RANK() OVER(PARTITION BY ile_tagow ORDER BY cnt) AS drank_cnt FROM t2
                    WHERE cnt > 0
            ) AS t
            WHERE """+str(sql_where_tags) +"""
            )
            SELECT id_informacji FROM t3
            """
        #print(sql)
        print(f'ZEROWA ITEREACJA')
        cursor = conn.cursor()
        cursor.execute(sql)
        data = cursor.fetchall()
        cursor.close()
        data = pd.DataFrame(data,columns=['id_informacji'])
        return data['id_informacji'].tolist()
    except (Exception, psycopg2.DatabaseError) as error:
        print("Error: %s" % error)
        conn.rollback()
        cursor.close()
        return None


@timeit
def get_data_embedding(model,
                       ls_field: list[str],
                       ls_text: list[str],
                       option_tpod: str,
                       keywords:list[str],
                       #query_add_words_to_db:list[str],
                       schema: str,
                       table_vector_emb: str,
                       table_info: str,
                       table_dict: str,
                       conn):
    try:
        schema_table_vec_emb = f'{schema}.{table_vector_emb}'
        schema_table_info = f'{schema}.{table_info}'
        schema_table_dict = f'{schema}.{table_dict}'
        query_to_db = f""" tpod IN ( '{option_tpod.upper()}' ) 
        AND rok BETWEEN extract(year from CURRENT_DATE)-3 AND extract(year from CURRENT_DATE) """
        query_keywords = calculate_query_list_words(keywords=keywords)
        comb_keywords = generate_combinations(word_list=keywords)
        sql_keywords = generate_sql_query(combinations_list=comb_keywords)

        ls_id_info_zero = get_zero_iter_from_db(conn=conn,
                          keywords=keywords,
                          query_to_db=query_to_db,
                          query_keywords=query_keywords,
                          sql_keywords=sql_keywords,
                          schema=schema, table=table_vector_emb)
        if ls_id_info_zero:
            #ls_id_info_zero = sorted(list(set(ls_id_info_zero)), reverse=True)
            ls_id_info_zero = ','.join([str(item) for item in ls_id_info_zero])
            ls_id_info_zero = f' id_informacji IN ('+ls_id_info_zero+') '
        else:
            ls_id_info_zero = f' 1=1 '

        ls_id_info = []
        for field, item in zip(ls_field,ls_text):

            print(f'PIERWSZA ITEREACJA: field - {field}')
            sql = """WITH t0 AS (
                SELECT id_informacji 
                FROM """ + str(schema_table_vec_emb) + """
                WHERE obszar IN ('sl_wartosc') 
                AND """ + str(query_to_db) + """
                AND """ + str(query_keywords) + """
                AND """ + str(ls_id_info_zero) + """
            ), """

            user_query_embedding = get_embedding_for_user_query(model, item[0])
            if field == 'doc_art' or field == 'doc_bigram':
                sql_tmp = """ t1 AS (
                SELECT id_informacji, 1 - (embedding <=> '""" + str(user_query_embedding) + """') AS cosinus_sim 
                FROM """ + str(schema_table_vec_emb) + """ AS ta 
                WHERE """ + str(query_to_db) + """ 
                AND EXISTS (SELECT 1 FROM t0 AS tb WHERE ta.id_informacji = tb.id_informacji )
                AND obszar IN ( 'sl_wartosc' ) 
                AND (1 - (embedding <=> '""" + str(user_query_embedding) + """'))> 0.97
                ), """
                sql = sql + sql_tmp
            elif field == 'doc_czy':
                sql_tmp = """ t1 AS (
                SELECT id_informacji, 1 - (embedding <=> '""" + str(user_query_embedding) + """') AS cosinus_sim 
                FROM """ + str(schema_table_vec_emb) + """ AS ta 
                WHERE 1=1
                AND """ + str(query_to_db) + """ 
                AND EXISTS (SELECT 1 FROM t0 AS tb WHERE ta.id_informacji = tb.id_informacji ) 
                AND obszar IN ( 'doc_czy' , 'teza_czy' ) 
                AND (LENGTH(opis) - LENGTH(replace(opis, ' ', '')))>5 
                AND (1 - (embedding <=> '""" + str(user_query_embedding) + """'))> 0.9
                ), """
                sql = sql + sql_tmp
            elif field == 'doc_podzal':
                sql_tmp = """ t1 AS (
                SELECT id_informacji, 1 - (embedding <=> '""" + str(user_query_embedding) + """') AS cosinus_sim 
                FROM """ + str(schema_table_vec_emb) + """ AS ta 
                WHERE 1=1 
                AND """ + str(query_to_db) + """ 
                AND EXISTS (SELECT 1 FROM t0 AS tb WHERE ta.id_informacji = tb.id_informacji ) 
                AND obszar IN ( 'doc_podzal' , 'teza_podzal' ) 
                AND (LENGTH(opis) - LENGTH(replace(opis, ' ', '')))>5 
                AND (1 - (embedding <=> '""" + str(user_query_embedding) + """'))> 0.9
                ), """
                sql = sql + sql_tmp
            elif field == 'doc_nie':
                sql_tmp = """ t1 AS (
                SELECT id_informacji, 1 - (embedding <=> '""" + str(user_query_embedding) + """') AS cosinus_sim 
                FROM """ + str(schema_table_vec_emb) + """ AS ta 
                WHERE 1=1 
                AND """ + str(query_to_db) + """
                AND EXISTS (SELECT 1 FROM t0 AS tb WHERE ta.id_informacji = tb.id_informacji ) 
                AND obszar IN ( 'doc_nie' ) 
                AND (LENGTH(opis) - LENGTH(replace(opis, ' ', '')))>5 
                AND (1 - (embedding <=> '""" + str(user_query_embedding) + """'))> 0.9
                ), """
                sql = sql + sql_tmp

            elif field == 'doc_slowa': 
                sql_tmp = """ t1 AS (
                SELECT id_informacji, 1 - (embedding <=> '""" + str(user_query_embedding) + """') AS cosinus_sim 
                FROM """ + str(schema_table_vec_emb) + """ AS ta 
                WHERE 1=1 
                AND """ + str(query_to_db) + """ 
                AND EXISTS (SELECT 1 FROM t0 AS tb WHERE ta.id_informacji = tb.id_informacji ) 
                AND obszar IN ( 'teza_czy' , 'teza_art','teza_podzal' , 'doc_podzal' ) 
                AND (LENGTH(opis) - LENGTH(replace(opis, ' ', '')))>5 
                AND (1 - (embedding <=> '""" + str(user_query_embedding) + """'))< 0.85
                AND (1 - (embedding <=> '""" + str(user_query_embedding) + """'))> 0.75
                ), """
                sql = sql + sql_tmp


            sql_tmp = """ t2 AS ( 
                SELECT id_informacji
                FROM (
                SELECT id_informacji, 
                    cosinus_sim, 
                    rank() over (partition by id_informacji order by cosinus_sim desc) as rank 
                FROM t1
                ) ta
                WHERE rank between 5 AND 25 
                GROUP BY id_informacji 
                ORDER BY AVG(cosinus_sim) DESC 
                LIMIT 20 ) 
                SELECT id_informacji FROM t2 """

            sql = sql + sql_tmp
            cursor = conn.cursor()
            cursor.execute(sql)
            for row in cursor.fetchall():
                #print(row)
                ls_id_info.append(tuple(row))
            cursor.close()

        ls_id_info = sorted(list(set(list(sum(ls_id_info,())))))
        if ls_id_info:
            ls_id_info = ','.join([str(item) for item in ls_id_info])
            ls_id_info = f' id_informacji IN ('+ls_id_info+') '
        else:
            return pd.DataFrame()

        data = []
        for field, item in zip(ls_field,ls_text):
            print(f'DRUGA ITEREACJA: field - {field}') 
            #print(item)
            user_query_embedding = get_embedding_for_user_query(model, item[0])
            sql_tmp = """ WITH t0 AS (
                SELECT id_informacji, obszar, 1 - (embedding <=> '""" + str(user_query_embedding) + """') AS cosinus_sim 
                FROM """ + str(schema_table_vec_emb) + """ AS ta 
                WHERE 1=1 
                AND """ + str(query_to_db) + """ 
                AND """ + str(ls_id_info) + """ 
                AND obszar IN ( 'sl_wartosc' , 'doc_art' , 'doc_czy' , 'teza_czy', 'doc_podzal' , 'teza_podzal' , 'doc_nie' ) 
                ), """
            sql_end = """ t1 AS ( 
                SELECT id_informacji, obszar, AVG(cosinus_sim) AS cosinus_sim, SUM(cosinus_sim) AS sum_cosinus_sim
                FROM (
                SELECT id_informacji,
                    cosinus_sim,
                    obszar,
                    rank() over (partition by id_informacji, obszar order by cosinus_sim desc) as rank 
                FROM t0
                ) ta
                WHERE rank <= 25 
                GROUP BY id_informacji, obszar 
                ),
                t2 AS (
                SELECT id_informacji, AVG(cosinus_sim) AS cosinus_sim FROM t1
                GROUP BY id_informacji
                ORDER BY SUM(sum_cosinus_sim) DESC
                LIMIT 20
                ),
                t3 AS (
                SELECT 
                    ta.id_informacji
                    , ta.tpod 
                    , CAST(ta.data_wydania AS date) AS data_wydania
                    , ta.sygnatura
                    , ta.teza
                FROM """ + str(schema_table_info) + """ AS ta 
                WHERE 1=1 
                AND """ + str(query_to_db) + """
                AND """ + str(ls_id_info) + """
                ), 
                t4 AS (
                SELECT DISTINCT tb.cosinus_sim, ta.tpod, ta.id_informacji, ta.data_wydania, ta.sygnatura, ta.teza 
                FROM t3 AS ta 
                INNER JOIN t2 AS tb 
                ON 1=1 
                AND ta.id_informacji = tb.id_informacji 
                ),
                t5 AS (
                SELECT 
                    id_informacji
                    , string_agg( DISTINCT wartosc , ' | ' ORDER BY wartosc) AS slowa_kluczowe 
                FROM (
                SELECT id_informacji, lower(wartosc) AS wartosc 
                FROM """ + str(schema_table_dict) + """ AS ta 
                WHERE 1=1 
                AND """ + str(query_to_db) + """
                AND """ + str(ls_id_info) + """
                AND dok_info IN ( 'czy_slowa_kluczowe' ) 
                ) AS ta
                GROUP BY id_informacji 
                )
                SELECT cosinus_sim, tpod, id_informacji, data_wydania, sygnatura, teza, slowa_kluczowe FROM t4
                LEFT JOIN t5
                USING (id_informacji)
                """


            sql = sql_tmp + sql_end
            cursor = conn.cursor()
            cursor.execute(sql)
            for row in cursor.fetchall():
                #print(row)
                data.append(tuple(row))
            cursor.close()

        #data = pd.DataFrame(data,columns=['id_informacji', 'cosinus_sim'])
        data = pd.DataFrame(data,columns=['cosinus_sim','tpod','id_informacji', 'data_wydania', 'sygnatura','teza','slowa_kluczowe'])
        data = data.drop_duplicates()
        data = data.sort_values(by='cosinus_sim', ascending=False)
#       return data
        print(data.head(1))

        return data

    except (Exception, psycopg2.DatabaseError) as error:
            print("Error: %s" % error)
            conn.rollback()
            cursor.close()
            return pd.DataFrame()




def store_db_text_input(conn,schema: str,
                        table: str, 
                        nr_uuid_sesji:str,
                        licznik_sesji: int,
                        znacznik_czasu_wpisu,
                        tekst_wpisu:str,
                        edycja_app: int,
                        hist_nr_uuid_sesji: str,
                        wybor_tpod: str
                        ):
    """ Funkcja """
    data = pd.DataFrame({'nr_uuid_sesji': [nr_uuid_sesji],
                        'licznik_sesji': [licznik_sesji],
                        'znacznik_czasu_wpisu': [znacznik_czasu_wpisu],
                        'tekst_wpisu': [tekst_wpisu],
                        'edycja_app': [edycja_app],
                        'hist_nr_uuid_sesji': [hist_nr_uuid_sesji],
                        'wybor_tpod': [wybor_tpod]
                        })

    schema_table = f'{schema}.{table}'

    tuples = [tuple(x) for x in data.to_numpy()]
    cols = ','.join(list(data.columns))
    # SQL query to execute
    query = "INSERT INTO %s(%s) VALUES %%s" % (schema_table, cols)
    cursor = conn.cursor()
    try:
        extras.execute_values(cursor, query, tuples)
        conn.commit()
    except (Exception, psycopg2.DatabaseError) as error:
        print("Error: %s" % error)
        conn.rollback()
        cursor.close()
        return 1
    print(f"the dataframe is inserted store_db_text_input: {schema_table}")
    cursor.close()


def store_db_score_input(conn,schema: str,
                         table: str,
                         nr_uuid_sesji: str,
                         licznik_sesji: int,
                         znacznik_ocena_wpisu, 
                         ocena_wpisu:str,
                         tekst_ocena_wpisu:str,
                         edycja_app: int,
                         hist_nr_uuid_sesji: str,
                         wybor_tpod: str
                         ):
    """ Funkcja """
    data = pd.DataFrame({'nr_uuid_sesji': [nr_uuid_sesji],
                        'licznik_sesji': [licznik_sesji],
                        'znacznik_ocena_wpisu': [znacznik_ocena_wpisu],
                        'ocena_wpisu': [ocena_wpisu],
                        'tekst_ocena_wpisu': [tekst_ocena_wpisu],
                        'edycja_app': [edycja_app],
                        'hist_nr_uuid_sesji': [hist_nr_uuid_sesji],
                        'wybor_tpod': [wybor_tpod]
                        })

    schema_table = f'{schema}.{table}'

    tuples = [tuple(x) for x in data.to_numpy()]
    cols = ','.join(list(data.columns))
    # SQL query to execute 
    query = "INSERT INTO %s(%s) VALUES %%s" % (schema_table, cols)
    cursor = conn.cursor()
    try: 
        extras.execute_values(cursor, query, tuples)
        conn.commit()
    except (Exception, psycopg2.DatabaseError) as error:
        print("Error: %s" % error)
        conn.rollback()
        cursor.close()
        return 1
    print(f"the dataframe is inserted store_db_score_input: {schema_table}")
    cursor.close()

def store_db_tab_output(conn,
                        schema: str,
                        table: str,
                        data: pd.DataFrame,
                        znacznik: str,
                        param_edycja_app: int,
                        nr_uuid_sesji: str,
                        hist_nr_uuid_sesji: str,
                        wybor_tpod:str
                        ):
    """ Funkcja """
    data = data[['cosinus_sim',
                 'id_informacji',
                 'data_wydania',
                 'sygnatura',
                 'teza',
                 'tpod']]
    data[['znacznik']] = znacznik
    data[['edycja_app']] = param_edycja_app
    data[['nr_uuid_sesji']] = nr_uuid_sesji
    data[['hist_nr_uuid_sesji']] = hist_nr_uuid_sesji
    data[['wybor_tpod']] = wybor_tpod

    schema_table = f'{schema}.{table}'

    tuples = [tuple(x) for x in data.to_numpy()]
    cols = ','.join(list(data.columns))
    # SQL query to execute 
    query = "INSERT INTO %s(%s) VALUES %%s" % (schema_table, cols)
    cursor = conn.cursor()
    try: 
        extras.execute_values(cursor, query, tuples)
        conn.commit()
    except (Exception, psycopg2.DatabaseError) as error:
        print("Error: %s" % error)
        conn.rollback()
        cursor.close()
        return 1
    print(f"the dataframe is inserted store_db_tab_output: {schema_table}")
    cursor.close()


if __name__ == '__main__':
    pass