
import os
from src.utils.dbsql import generate_sql_files_for_table, read_sql_files_for_table, create_table_db
import src.utils.logger as logger_utils
from dotenv import load_dotenv
load_dotenv()


###################################
TABLE_NAME_INTERP = os.environ.get("TBL_INTERP")
EU_SQL = os.environ.get("EU_SQL")
POSTGRES_DSN = os.environ.get("POSTGRES_DSN")

###################################
logger = logger_utils.setup_logger(name='eu_sql_interp')
logger.info(f"EU_SQL path: {EU_SQL}")

###################################
INTERP_DICT = {
    'id': 'SERIAL PRIMARY KEY',
    'id_informacji': 'INTEGER',
    'kategoria_informacji': 'INTEGER',
    'status_informacji': 'INTEGER',
    'data_publikacji': 'TIMESTAMP without time zone',
    'teza': 'TEXT',
    'dt_wyd': 'TIMESTAMP without time zone',
    'syg': 'TEXT',
    'slowa_kluczowe': 'TEXT',
    'przepisy': 'TEXT',
    'zagadnienia': 'TEXT',
    'tresc_interesariusz': 'TEXT',
    'tresc_interesariusz_html': 'TEXT',
    'id_eu': 'INTEGER[]',
    'kod_eu': 'TEXT[]',
    'opis_eu': 'TEXT[]',
    'wartosc_eu': 'TEXT[]',
    'przepisy_opis_eu': 'TEXT[]',
    'przepisy_wartosc_eu': 'TEXT[]',
    'zagadnienia_opis_eu': 'TEXT[]',
    'zagadnienia_wartosc_eu': 'TEXT[]',
    'slowa_kluczowe_opis_eu': 'TEXT[]',
    'slowa_kluczowe_wartosc_eu': 'TEXT[]',
    'typ_podatku': 'TEXT',
    'typ_podatku_opis': 'TEXT',
    'versionid': 'TEXT',
    'nazwa': 'TEXT',
    'szablonid': 'SMALLINT',
    'wersjaszablonuid': 'SMALLINT',
    'dataczas_akt': 'TIMESTAMP without time zone'
}

generate_sql_files_for_table(table_name=TABLE_NAME_INTERP, schema_dict=INTERP_DICT, output_dir=EU_SQL)
READ_SQL = read_sql_files_for_table(table_name=TABLE_NAME_INTERP, input_dir=EU_SQL)
CREATE_SQL = READ_SQL.get('create_sql',[])
INSERT_SQL = READ_SQL.get('insert_sql',[])
COLUMNS = READ_SQL.get('columns_sql',[])

create_table_db(connection_string=POSTGRES_DSN, sql=CREATE_SQL)

logger.info(f'CREATE sql files for table {CREATE_SQL}')
logger.info(f'INSERT sql files for table {INSERT_SQL}')
logger.info(f"COLUMNS from EU_SQL: {COLUMNS}")