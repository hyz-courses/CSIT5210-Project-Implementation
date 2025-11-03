import os
from typing import cast, Optional

from loguru import logger as _logger

import numpy as np

from data_process import JsonLoader, NpyLoader
from run_LLM.encoder import llm2vec_encoder_factory
from utils.logs import bind_logger

THIS_FILE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT_DIR = os.path.join(THIS_FILE_DIR, "..")

logger = bind_logger(_logger,
                     log_path=os.path.join(
                        THIS_FILE_DIR, "..",
                        "logs", "extract_embedding.log"))

def extract_category_embedding(
        category: str,
        model_path: str,
        peft_model_path: Optional[str],
        batch_size: int,
        bidirectional: bool,
        instruction: Optional[str] = None):
    
    """
    Extract item title embeddings for a given category.
    """
    
    title2id = JsonLoader(
        category=category,
        phase="downstream",
        usage="title2id",
        project_root=PROJECT_ROOT_DIR
    ).load()

    item_ids = title2id.values()

    if any([int(x) == 0 for x in item_ids]):
        raise ValueError(
            "Item ID should not be 0, as 0 is reserved for NULL object.")

    item_titles = ["NULL"] + list(title2id.keys())
    item_titles = np.array(item_titles)

    if instruction is not None:
        instruction_rep = np.repeat(instruction, len(item_titles))
        prompts = np.concatenate((
            instruction_rep[:, np.newaxis],
            item_titles[:, np.newaxis]
        ), axis=1)
    else:
        prompts = item_titles
    
    model = llm2vec_encoder_factory(
        base_model_name_or_path=model_path,
        peft_model_name_or_path=peft_model_path,
        bidirectional=bidirectional)
    
    item_embeddings = model.encode(
        list(prompts), batch_size=batch_size, 
        convert_to="numpy")
    
    item_embeddings = cast(np.ndarray, item_embeddings)
    
    NpyLoader(
        category=category,
        phase="downstream",
        usage="emb",
        project_root=PROJECT_ROOT_DIR
    ).store(item_embeddings)

    
if __name__ == "__main__":
    outofdomain_categories = ["Baby_Products", "Sports_and_Outdoors"]

    model_path_checkpoint1000 = os.path.join(
        PROJECT_ROOT_DIR, "output", "iem_stage2", 
        "Qwen2-0.5B-CSFT-AmazonMix-CSIT5210G1", "checkpoint-1000")

    for cat in outofdomain_categories:
        extract_category_embedding(
            category=cat,
            model_path=model_path_checkpoint1000,
            peft_model_path=None,
            batch_size=32,
            bidirectional=True,
            instruction=None)
    


