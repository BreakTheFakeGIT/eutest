
import os
from src.utils.dbsql import generate_sql_files_for_table, read_sql_files_for_table, create_table_db
import src.utils.logger as logger_utils
from dotenv import load_dotenv
load_dotenv()


###################################
TABLE_NAME_TEXT_SPLIT = os.environ.get("TBL_TEXT_SPLIT")
EU_SQL = os.environ.get("EU_SQL")
POSTGRES_DSN = os.environ.get("POSTGRES_DSN")

###################################
logger = logger_utils.setup_logger(name='eu_sql_text_split')
logger.info(f"EU_SQL path: {EU_SQL}")

###################################
TEXT_SPLIT_DICT = {
    'id': 'SERIAL PRIMARY KEY',
    'eksperyment': 'VARCHAR(255)',
    'tresc_fragment_id': 'INTEGER',
    'tresc_fragment': 'TEXT',
    'dataczas_eks': 'TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP',
    'id_informacji': 'INTEGER',
    'kategoria_informacji': 'INTEGER',
    'status_informacji': 'INTEGER',
    'teza': 'TEXT',
    'dt_wyd': 'TIMESTAMP without time zone',
    'syg': 'TEXT',
    'tresc_interesariusz': 'TEXT',
    'przepisy_opis_eu': 'TEXT[]',
    'przepisy_wartosc_eu': 'TEXT[]',
    'slowa_kluczowe_wartosc_eu': 'TEXT[]',
    'typ_podatku': 'TEXT',
    'typ_podatku_opis': 'TEXT',
    'dataczas_akt': 'TIMESTAMP without time zone'
}

generate_sql_files_for_table(table_name=TABLE_NAME_TEXT_SPLIT, schema_dict=TEXT_SPLIT_DICT, output_dir=EU_SQL)
READ_SQL = read_sql_files_for_table(table_name=TABLE_NAME_TEXT_SPLIT, input_dir=EU_SQL)
CREATE_SQL = READ_SQL.get('create_sql',[])
INSERT_SQL = READ_SQL.get('insert_sql',[])
COLUMNS = READ_SQL.get('columns_sql',[])

create_table_db(connection_string=POSTGRES_DSN, sql=CREATE_SQL)

logger.info(f'CREATE sql files for table {CREATE_SQL}')
logger.info(f'INSERT sql files for table {INSERT_SQL}')
logger.info(f"COLUMNS from EU_SQL: {COLUMNS}")


