-- Run once
-- CREATE EXTENSION IF NOT EXISTS vector;
-------------------------------------------------------------------------
-- qa_result_id BIGINT NOT NULL REFERENCES qa_results(id) !!!!!!!!!!!!!!

DROP TABLE answer_embeddings_qa
DROP TABLE answer_embeddings_sa
DROP TABLE qa_results --CASCADE
DROP TABLE stat_results

TRUNCATE TABLE qa_results CASCADE
TRUNCATE TABLE answer_embeddings_qa
TRUNCATE TABLE answer_embeddings_sa
TRUNCATE TABLE stat_results


--TRUNCATE qa_results RESTART IDENTITY




DELETE FROM qa_results
WHERE text_id = 669825

--DELETE FROM answer_embeddings
--WHERE text_id = 669935


-------------------------------------------------------------------------
-- Statistics
-------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS stat_results (
  id BIGSERIAL PRIMARY KEY,
  text_id INTEGER NOT NULL,
  tax_type TEXT NOT NULL,
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
  keywords TEXT[] NOT NULL,
  created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP NOT NULL,
  updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP NOT NULL 
);

-------------------------------------------------------------------------
-- Embeddings of answers for retrieval/RAG
CREATE TABLE IF NOT EXISTS answer_embeddings_qa (
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

CREATE TABLE IF NOT EXISTS answer_embeddings_sa (
  id BIGSERIAL NOT NULL PRIMARY KEY,
  qa_results_id BIGINT NOT NULL,
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
SELECT * FROM answer_embeddings_qa
SELECT * FROM answer_embeddings_sa
SELECT * FROM stat_results



SELECT 
text_id,
tax_type,
json_extract_path_text(results,'chunks') AS chunks, 
json_extract_path_text(results,'text_norm_len') AS text_norm_len ,
json_extract_path_text(results,'timings', 'llm_latency_s') AS llm_latency_s ,
json_extract_path_text(results,'timings', 'total_s') AS total_s  
FROM stat_results


-------------------------------------------------------------------------
{"text_id": 669935, "tax_type": "vat", "chunks": 5, "text_norm_len": 68743, "questions_per_context": 14, 
"llm_models": ["hf.co/NikolayKozloff/Llama-PLLuM-8B-instruct-Q8_0-GGUF:Q8_0"], "embed_models": "/dane/models/sdadas/stella-pl-retrieval", "embed_dim": 1024, "asyncio_semaphore": 10, "qa_results_saved": 70, 
"timings": {"chunking_s": 0.002, "llm_latency_s": 1.886, "qa_llm_s": 45.323, "total_s": 45.784}}
{"text_id": 669933, "tax_type": "pit", "chunk_id": 1, "question_id": 1, "chunk_text_len": 1997, "answer_len": 24, "text_norm_len": 8868, "questions_per_context": 11, "llm_models": ["hf.co/NikolayKozloff/Llama-PLLuM-8B-instruct-Q8_0-GGUF:Q8_0"], "embed_models": "/dane/models/sdadas/stella-pl-retrieval", "embed_dim": 1024, "asyncio_semaphore": 10, 
"timings": {"chunking_s": 0.001, "llm_latency_s": 8.82}}

--DELETE FROM qa_results
--WHERE text_id = 669825
--SELECT * FROM answer_embeddings
--WHERE text_id = 669825


-------------------------------------------------------------------------
SELECT 
    id_informacji,
    typ_podatku,
    tresc_interesariusz,
    TO_CHAR(dt_wyd, 'YYYY-MM-DD') AS dt_wyd,
    syg,
    teza,
    slowa_kluczowe_wartosc_eu,
    wartosc_eu,
    MIN(id_informacji) OVER() AS min_id_informacji, -- NOT EXISTS IN pyhton code
    MAX(id_informacji) OVER() AS max_id_informacji  -- NOT EXISTS IN pyhton code
FROM public.interpretacje AS ta
WHERE True
    AND kategoria_informacji = 1
    AND szablonid IN (1,2)
    AND typ_podatku IN ('vat','pit','cit','pcc','psd',
    'akcyza','op','gry','malpki','spw','pt','pkop',
    'spdet','fin','cukier','wip','globe','nip','inne')
    AND NOT EXISTS (
        SELECT 1
        FROM public.qa_results AS qa 
        WHERE qa.text_id = ta.id_informacji
    )
    AND NOT EXISTS (
        SELECT 1
        FROM public.stat_results AS qb
        WHERE qb.text_id = ta.id_informacji
    )
ORDER BY id_informacji DESC 

----------------------------------------------------------------------------------
----------------------------------------------------------------------------------
----------------------------------------------------------------------------------
-- TEST -- TEST -- TEST -- TEST -- TEST -- TEST -- TEST -- TEST -- TEST -- TEST 
----------------------------------------------------------------------------------
----------------------------------------------------------------------------------
DROP TABLE qa_results_test CASCADE
DROP TABLE answer_embeddings_qa_test
DROP TABLE answer_embeddings_sa_test
DROP TABLE stat_results_test

TRUNCATE TABLE qa_results_test CASCADE
TRUNCATE TABLE answer_embeddings_qa_test
TRUNCATE TABLE answer_embeddings_sa_test
TRUNCATE TABLE stat_results_test



-- QA results per chunk/question/model
CREATE TABLE IF NOT EXISTS qa_results_test (
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
  keywords TEXT[] NOT NULL,
  created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP NOT NULL,
  updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP NOT NULL 
);

-------------------------------------------------------------------------
-- Embeddings of answers for retrieval/RAG
CREATE TABLE IF NOT EXISTS answer_embeddings_qa_test (
  id BIGSERIAL NOT NULL PRIMARY KEY,
  qa_results_id BIGINT NOT NULL REFERENCES qa_results_test(id) ON DELETE CASCADE,
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

CREATE TABLE IF NOT EXISTS answer_embeddings_sa_test (
  id BIGSERIAL NOT NULL PRIMARY KEY,
  qa_results_id BIGINT NOT NULL,
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


CREATE TABLE IF NOT EXISTS stat_results_test (
  id BIGSERIAL PRIMARY KEY,
  text_id INTEGER NOT NULL,
  tax_type TEXT NOT NULL,
  type_results TEXT NOT NULL, -- a piece of python code
  results JSON NOT NULL, -- in general timings in seconds !!! (_s)
  created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP NOT NULL,
  updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP NOT NULL
);



SELECT * FROM qa_results_test 
WHERE question_id = 4
ORDER BY id DESC


SELECT * FROM answer_embeddings_qa_test ORDER BY id DESC

INSERT INTO  answer_embeddings_sa_test


SELECT * FROM answer_embeddings_qa_test
SELECT * FROM answer_embeddings_sa_test




SELECT * FROM stat_results_test ORDER BY id DESC



WITH t0 AS (
SELECT * FROM answer_embeddings_qa_test
WHERE type_text='chunk' 
)

INSERT INTO answer_embeddings_sa_test
SELECT * FROM t0 AS ta
WHERE qa_results_id = ANY(SELECT MIN(qa_results_id) 
FROM t0 AS tb WHERE ta.text_id = tb.text_id)


SELECT * FROM answer_embeddings_sa_test



