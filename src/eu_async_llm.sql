-- Run once
-- CREATE EXTENSION IF NOT EXISTS vector;
-------------------------------------------------------------------------
-- qa_result_id BIGINT NOT NULL REFERENCES qa_results(id) !!!!!!!!!!!!!!

DROP TABLE answer_embeddings
DROP TABLE qa_results --CASCADE
DROP TABLE stat_results

TRUNCATE TABLE answer_embeddings
TRUNCATE TABLE qa_results CASCADE
TRUNCATE TABLE stat_results

--DELETE FROM answer_embeddings
--WHERE text_id = 669935
--DELETE FROM qa_results
--WHERE text_id = 669935

-------------------------------------------------------------------------
-- Statistics
-------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS stat_results (
  id BIGSERIAL PRIMARY KEY,
  type_results TEXT NOT NULL, -- a piece of python code
  results JSON NOT NULL, -- in general timings in seconds !!! (_s)
  created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP NOT NULL,
  updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP NOT NULL
);


-------------------------------------------------------------------------
-- QA results per chunk/question/model
CREATE TABLE IF NOT EXISTS qa_results (
  id BIGSERIAL NOT NULL PRIMARY KEY,
  text_id INTEGER NOT NULL, -- id_informacji
  tax_type TEXT NOT NULL,
  chunk_id SMALLINT NOT NULL,
  question_id SMALLINT NOT NULL,
  model_id SMALLINT NOT NULL,
  model_name TEXT NOT NULL,
  chunk_text TEXT NOT NULL, 
  question TEXT NOT NULL,
  answer TEXT NOT NULL, 
  is_excluded SMALLINT NOT NULL, 
  llm_latency_ms INTEGER NOT NULL, -- timing in milliseconds !!! (_ms)
  created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP NOT NULL,
  updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP NOT NULL 
);

-------------------------------------------------------------------------
-- Embeddings of answers for retrieval/RAG
CREATE TABLE IF NOT EXISTS answer_embeddings (
  id BIGSERIAL NOT NULL PRIMARY KEY,
  qa_results_id BIGINT NOT NULL REFERENCES qa_results(id) ON DELETE CASCADE,
  text_id INTEGER NOT NULL,
  tax_type TEXT NOT NULL,
  question_id SMALLINT NOT NULL,
  show_in_chat JSON NOT NULL,
  is_excluded SMALLINT NOT NULL,
  type_text TEXT NOT NULL,
  embedding VECTOR(1024) NOT NULL,
  created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP NOT NULL,
  updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP NOT NULL
);

-------------------------------------------------------------------------
--CREATE INDEX IF NOT EXISTS idx_answer_embeddings_cosine
--  ON answer_embeddings USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
--ANALYZE answer_embeddings;
-------------------------------------------------------------------------


SELECT * FROM qa_results 
SELECT * FROM answer_embeddings
SELECT * FROM stat_results

{"fetched": 3, "timings": {"fetch_s": 0.2516274629160762, "process_s": 47.16695692203939}, 
"contexts": [
{"text_id": 669935, "tax_type": "vat", "chunks": 6, "questions_per_context": 5, "models": ["hf.co/NikolayKozloff/Llama-PLLuM-8B-instruct-Q8_0-GGUF:Q8_0"], "qa_results_saved": 30, "timings": {"chunking_s": 0.008742337115108967, "llm_latency_s": 1.721, "qa_llm_s": 13.914772531017661, "total_s": 14.177183602936566}}, 
{"text_id": 669835, "tax_type": "vat", "chunks": 6, "questions_per_context": 5, "models": ["hf.co/NikolayKozloff/Llama-PLLuM-8B-instruct-Q8_0-GGUF:Q8_0"], "qa_results_saved": 30, "timings": {"chunking_s": 0.002523954026401043, "llm_latency_s": 6.044, "qa_llm_s": 46.88772740820423, "total_s": 47.034904941916466}}, 
{"text_id": 669788, "tax_type": "vat", "chunks": 6, "questions_per_context": 5, "models": ["hf.co/NikolayKozloff/Llama-PLLuM-8B-instruct-Q8_0-GGUF:Q8_0"], "qa_results_saved": 30, "timings": {"chunking_s": 0.0011457977816462517, "llm_latency_s": 5.132, "qa_llm_s": 31.062264129985124, "total_s": 31.129930967930704}
}]}
-------------------------------------------------------------------------











