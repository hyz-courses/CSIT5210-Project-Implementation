import os
import json
from pathlib import Path
from typing import List, DefaultDict, Callable
from tqdm import tqdm
import pickle
from loguru import logger

logger.add("logs/data_process.log", rotation="10 MB")

def load_jsonl_lines(
        func: Callable, 
        jsonl_file: str, 
        clip: int = -1) -> List[dict]:
    """
    Load a .jsonl file into a line of dictionaries.
    Parameters:
        func (callable): Processing lambda function for each record.
        jsonl_file (str): The path to the jsonl file.
        clip (int): The number of lines to be loaded.
                    Default, load all. For testing, just load a few samples. 
                    If it is set negative, then load all.
    Returns:
        List[dict]: A list of dictionaries.
    """

    if not jsonl_file.endswith(".jsonl"):
        message = (f"[CSIT5210 Err]: The file path you inputted" 
                    " is not a valid jsonl file.")
        logger.error(message)
        raise ValueError(message)

    if not os.path.exists(jsonl_file):
        message = (f"[CSIT5210 Err]: The file"
                   f" {jsonl_file} does not exist.")

        logger.error(message)
        raise FileNotFoundError(message)
    
    logger.info(
        f"[CSIT5210 Info]: Start loading jsonl file" 
        f" {Path(jsonl_file).stem}.jsonl.")

    dict_lines = []
    with open(jsonl_file, "r") as file_lines:
        for i, line in tqdm(enumerate(file_lines)):
            dict_lines.append(func(json.loads(line)))
            if clip > 0 and i >= clip:
                break
        file_lines.close()
    return dict_lines


def load_raw_data(category: str):
    """
    Load key contents of review and meta.
    Parameters:
        category (str): The category of the dataset.
                        E.g., "Video_Games", "Books", etc.
    """

    base_path = "data/raw"
    review_file, meta_file = f"{category}.jsonl", f"meta_{category}.jsonl"

    meta_list = load_jsonl_lines(
        func=lambda record: {
            "parent_asin": record["parent_asin"], 
            "title": record["title"]}, 
        jsonl_file=os.path.join(base_path, category, meta_file))
    
    review_list = load_jsonl_lines(
        func=lambda record: {
            "timestamp": record["timestamp"],
            "user_id": record["user_id"], 
            "asin": record["asin"],
            "parent_asin": record["parent_asin"]}, 
        jsonl_file=os.path.join(base_path, category, review_file))

    print(review_list[0])

    return review_list, meta_list


def load_user_interactions(
        reviews: List[dict], 
        metas: List[dict]):
    
    """
    Load user interactions.
    """
    
    logger.info(
        f"[CSIT5210 Info]: Loading user interactions.")

    itemid_itemtitle = {
        meta["parent_asin"]: meta["title"] for meta in metas}

    user_interactions = DefaultDict(list)
    for review in tqdm(reviews):
        user_interactions[review["user_id"]].append({
            "timestamp": review["timestamp"],
            "asin": review["asin"],
            "parent_asin": review["parent_asin"],
            "item_title": itemid_itemtitle[review["parent_asin"]]
        })

    return user_interactions


def save_user_interactions(
        user_interactions: DefaultDict[str, list],
        category: str):
    
    """
    Save user interactions to .jsonl and .pkl files.
    """
    
    base_path = "data/grained"
    category_path = os.path.join(base_path, category)

    if not os.path.exists(category_path):
        os.makedirs(category_path)

    logger.info(
        f"[CSIT5210 Info]: Loading user interactions to"
        " .jsonl (for visualization).")
    
    with open(os.path.join(category_path, f"{category}_usritn.jsonl"), "w") as f:
        json.dump(user_interactions, f)
        f.close()

    logger.info(
        f"[CSIT5210 Info]: Loading user interactions to pickle" 
        " (for loading).")
    
    with open(os.path.join(category_path, f"{category}_usritn.pkl"), "wb") as f:
        pickle.dump(user_interactions, f)
        f.close()

if __name__ == "__main__":
    reviews, metas = load_raw_data(category="Video_Games")
    uits = load_user_interactions(reviews, metas)
    save_user_interactions(uits, category="Video_Games")

    # with open("data/grained/Video_Games/Video_Games_usritn.pkl", "rb") as f:
    #     uits = pickle.load(f)
    #     for k, v in uits.items():
    #         print(k)
    #         print(v)
    #     f.close()
    
