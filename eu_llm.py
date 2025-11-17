from typing import Dict, Any, Optional
from sentence_transformers import SentenceTransformer
from langchain_ollama.llms import OllamaLLM
from src.prompts.prompt_taxes import TAX_PROMPTS

import src.utils.logger as logger_utils
from dotenv import load_dotenv
load_dotenv()

# --- Konfiguracja logowania ---
logger = logger_utils.setup_logger(name=f'eu_llm')

# --- 1. Przypisanie promptu (w języku polskim) ---


def process_tax_text(
    user_text: str,
    tax_type: str,
    # conn: psycopg.Connection,
    # llm: OllamaLLM,
    # embedding_model: SentenceTransformer
) -> Optional[Dict[str, Any]]:
    """"""
    logger.info(f"Rozpoczynam przetwarzanie dla typu: {tax_type}")

    # --- 1. Przypisanie promptu ---
    prompt_template = TAX_PROMPTS.get(tax_type, TAX_PROMPTS["inne"])
    formatted_prompt = prompt_template.format(user_text=user_text)
    print(formatted_prompt)
    logger.info("Prompt został przygotowany.")




    #logger.info(f"Zakończono przetwarzanie dla ID: {new_id}.")
    #return results_dict





# --- Przykładowe użycie ---
def main():
    user_text = "Główna funkcja do uruchomienia przykładu."
    tax_type = "cit"
    process_tax_text(user_text=user_text,tax_type=tax_type)
    # # --- Inicjalizacja modeli ---
    # try:
    #     logger.info("Inicjalizacja modelu LLM (Ollama)...")
            llm = OllamaLLM(model="bielik") 
            llm.invoke("test") 
            logger.info("Model LLM gotowy.")

    #     logger.info("Inicjalizacja modelu SentenceTransformer...")
    #     embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    #     # ZMIANA: Pobierz wymiar wektora z modelu
    #     dimension = embedding_model.get_sentence_embedding_dimension()
    #     logger.info(f"Model SentenceTransformer gotowy. Wymiar wektora: {dimension}")

    # except Exception as e:
    #     logger.error(f"Błąd podczas inicjalizacji modeli: {e}")
    #     logger.error("Upewnij się, że Ollama jest uruchomiona i masz model 'bielik'.")
    #     logger.error("Upewnij się, że biblioteka 'sentence-transformers' jest zainstalowana.")
    #     return

    # # --- Połączenie z bazą danych (Psycopg 3) ---
    # DB_CONN_STRING = "dbname=postgres user=postgres password=mysecretpassword host=localhost port=5432"


if __name__ == "__main__":
    main()


