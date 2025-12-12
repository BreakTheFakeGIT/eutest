'''import model from Hugging Face'''

import os
from sentence_transformers import SentenceTransformer
import torch
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from transformers import AutoProcessor, AutoModelForCausalLM
#from transformers import AutoFeatureExtractor, AutoModelForObjectDetection
from IPython.display import clear_output
import src.utils.logger as logger_utils
from typing import List
from dotenv import load_dotenv
load_dotenv()

###################################
logger = logger_utils.setup_logger(name=f'eu_load_hf')

###################################
PATH_MODELS = os.environ.get("MODELS_HF")

def make_dir_path(path: str):
    '''Function'''
    if os.path.isdir(path) == False: os.makedirs(path)
    logger.info(f'Create folder: {path}')

def model_from_hf(path_models: str, list_names_models: List[str], version: str):
    '''Function'''
    for name_model in list_names_models:
        path_model = f'{path_models}{name_model}'

        if os.path.exists(path_model) and len(os.listdir(path_model)) > 0:
            logger.info(f'Exists: {path_model}')
            continue
        else:
            os.remove(path_model)
            logger.info(f'Remove folders: {os.listdir(path_model)}')

        make_dir_path(path=path_model)

        if version == 'SentenceTransformer':
            model = SentenceTransformer(name_model, trust_remote_code=True, device="cuda")
            model.save(path_model)
            logger.info(f"Save model.save {name_model}. Version {version}. Save path  {path_model}")

        elif version == 'TrOCRProcessor' or version == 'AutoProcessor':
            match version:
                case 'TrOCRProcessor':
                    #'microsoft/trocr-large-handwritten'
                    processor = TrOCRProcessor.from_pretrained(name_model)
                    model = VisionEncoderDecoderModel.from_pretrained(name_model)
                case 'AutoProcessor':
                    ## 'yifeihu/TF-ID-large-no-caption' Run `pip install timm flash_attn einops` problem z wheel nie tykaccleas
                    processor = AutoProcessor.from_pretrained(name_model, trust_remote_code=True)
                    model = AutoModelForCausalLM.from_pretrained(name_model, trust_remote_code=True)

            processor.save_pretrained(path)
            model.save_pretrained(path)
            logger.info(f"Save model.save_pretrained {name_model}. Version {version}. Save path  {path_model}")

        else:
            logger.error('Not save model')

    clear_output(wait=True)


if __name__== '__main__':

    list_models = ['sdadas/stella-pl-retrieval',
    'sdadas/stella-pl',
    'sdadas/mmlw-retrieval-roberta-large-v2',
    'all-MiniLM-L6-v2',
    'sdadas/st-polish-paraphrase-from-mpnet',
    'sdadas/mmlw-roberta-large',
    'sdadas/mmlw-retrieval-roberta-large',
    'sdadas/polish-splade',
    'sdadas/st-polish-paraphrase-from-distilroberta',
    'sdadas/mmlw-e5-large',
    'sdadas/polish-longformer-base-4096',
    'ipipan/silver-retriever-base-v1.1',
    "BAAI/bge-multilingual-gemma2",
    "infly/inf-retriever-v1"
    ]


    model_from_hf(path_models=PATH_MODELS, list_names_models=list_models, version='SentenceTransformer')
