import os
import json
from pathlib import Path
from typing import List, DefaultDict, Callable

# Data processing
import pandas as pd
import csv

from tqdm import tqdm
from loguru import logger



logger.add("logs/data_process.log", rotation="10 MB")

"""
Data Processing Infrastructures
"""

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


def extend_csv_file(dict_list: List[dict], category: str):
    with open(f"data/grained/{category}/{category}_userint.csv") as csv_file:
        field_names = dict_list[0].keys()
        writer = csv.DictWriter(csv_file, fieldnames=field_names)
        writer.writeheader()
        for row in dict_list:
            for key, value in row.items():
                write_value = value
                if isinstance(value, list):
                    write_value = str(write_value)
                writer.writerow({key: write_value})

"""
Data Graining
"""


def load_raw_data(category: str):
    """
    Load key contents of review and meta.
    Parameters:
        category (str): 
            The category of the dataset.
            E.g., "Video_Games", "Books", etc.
    Returns:
        review_list (List[dict]): 
            A list of purified review records.
        meta_list (List[dict]): 
            A list of purified meta records.
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

    return review_list, meta_list


def get_pruned_user_interaction_list(reviews: List[dict]):
    """
    Get pruned list of user interactions, including:
        - user_id: Which user
        - timestamp: When the interaction happened
        - parent_asin: Which item
    Iteratively prune the list using 5-core.
        
    Parameters:
        reviews (List[dict]): 
            A list of review records.
    Returns:
        List[dict]: 
            A list of pruned user interaction records.
    """

    # which user interacted which item at which time
    records = [{
        "user_id": review["user_id"],
        "timestamp": review["timestamp"],
        "parent_asin": review["parent_asin"],
        } for review in reviews]

    df = pd.DataFrame(records)

    logger.info(
        f"[CSIT5210 Info]: Before filtering, \n\n"
        f"{len(df)} interactions, " 
        f"{df['user_id'].nunique()} distinct users, "
        f"{df['parent_asin'].nunique()} distinct items.\n\n")

    while True:
        user_counts = df["user_id"].value_counts()
        item_counts = df["parent_asin"].value_counts()

        # Remove users with less than 5 interactions
        valid_users = user_counts[user_counts >= 5].index

        # Remove items with less than 5 interactions
        valid_items = item_counts[item_counts >= 5].index

        df_new = df[
            df["user_id"].isin(valid_users) &
            df["parent_asin"].isin(valid_items)].copy()
        
        if len(df_new) == len(df):
            break

        df = df_new
    
    logger.info(
        f"[CSIT5210 Info]: After filtering, \n\n"
        f"{len(df)} interactions, " 
        f"{df['user_id'].nunique()} distinct users, "
        f"{df['parent_asin'].nunique()} distinct items.\n\n")
    
    return df


def build_asin_itemid_map(ui_df: pd.DataFrame):
    """
    Obtaining a pruned user interaction list,
    assign each item parent_asin a unique
    numerical item id, and organize a hashmap.

    Parameters:
        ui_df (pd.DataFrame): 
            A dataframe of user interactions.
    Returns:
        asin_to_itemid (dict): 
            A dictionary mapping parent_asin to item_id.
    """
    
    unique_asins = sorted(ui_df['parent_asin'].unique())
    asin_to_itemid = {asin: i for i, asin in enumerate(unique_asins)}

    return asin_to_itemid


# def grain_user_interaction_list(
#         ui_df: pd.DataFrame, 
#         asin_to_itemid: dict):
#     ui_df["item_id"] = ui_df["parent_asin"].map(asin_to_itemid)
#     ui_df = ui_df.sort_values(["user_id", "timestamp"])

#     user_itemlist = ui_df.groupby("user_id")["item_id"]


if __name__ == "__main__":
    reviews, metas = load_raw_data(category="Video_Games")

    df = get_pruned_user_interaction_list(reviews)
    map = build_asin_itemid_map(df)

    for k, v in map.items():
        print(f"{k} {v}")