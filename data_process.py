"""
CSIT5210 - Data Mining and Knowledge Discovery
@author: HUANG, Yanzhen
@date: Oct. 10, 2025
@description: Data processing infrastructure.
"""

# Basics
import os
import json
import math
from typing import List, Callable, TypeVar, Generic, Tuple, Optional, DefaultDict
from abc import ABC, abstractmethod
from datasets import Dataset, DatasetDict

# Data processing
import pandas as pd

# Console logs
from tqdm import tqdm
from loguru import logger


logger.add("logs/data_process.log", rotation="10 MB")
T = TypeVar("T")

"""
Data Processing Infrastructures
"""


class CategoryLoader(ABC, Generic[T]):
    """
    Loads dataset from a specific category.
    Parameters:
        category (str):
            The category of the dataset.
            E.g., "Video_Games", "Arts_Crafts_and_Sewing", etc.
        ext (str):
            The extension of the dataset file.
            E.g., "jsonl", "csv", etc.
        phase (str):
            The phase of the dataset.
            E.g., "raw", "grained".
        usage (str):
            The usage of the dataset.
            E.g., "meat", "train", "valid", "test".
    """

    def __init__(
        self,
        category: str,
        ext: str,
        phase: str,
        usage: str,
        limit: Optional[int] = None,
    ):
        self.ext = ext
        self.category = category
        self.type = usage
        self.phase = phase
        self.type = usage
        self.limit = limit

        if limit and limit < 0:
            message = (
                f"[CSIT5210 Err]: The limit you inputted " f"is invalid. ({limit})"
            )
            logger.error(message)
            raise ValueError(message)

    @abstractmethod
    def _load(self, file_path, func: Callable = lambda x: x) -> T:
        pass

    def __call__(self, func: Callable = lambda x: x) -> T:

        file_name = f"{self.category}.{self.ext}"

        if self.type != "":
            file_name = f"{self.type}_{file_name}"

        file_path = os.path.join("data", self.phase, self.category, file_name)

        if not file_path.endswith(f".{self.ext}"):
            message = (
                f"[CSIT5210 Err]: The file path you inputted "
                f"is not a valid .{self.ext} file. ({file_path})"
            )
            logger.error(message)
            raise ValueError(message)

        if not os.path.exists(file_path):
            message = f"[CSIT5210 Err]: The file" f" {file_path} does not exist."
            logger.error(message)
            raise FileNotFoundError(message)

        return self._load(file_path, func)


class JsonlLoader(CategoryLoader[List[dict]]):
    """
    Loads raw dataset in .jsonl format from a
    specific category.

    Parameters:
        category (str):
            The category of the dataset.
            E.g., "Video_Games", "Arts_Crafts_and_Sewing", etc.
        phase (str):
            The phase of the dataset.
            E.g., "raw", "grained".
        usage (str):
            The usage of the dataset.
            E.g., "meat", "train", "valid", "test".
    """

    def __init__(
        self, category: str, phase: str, usage: str = "", limit: Optional[int] = None
    ):
        super().__init__(
            category=category, ext="jsonl", phase=phase, usage=usage, limit=limit
        )

    def _load(self, file_path, func: Callable = lambda x: x) -> List[dict]:
        dict_lines = []
        with open(file_path, "r", encoding="utf-8") as file_lines:
            for i, line in tqdm(enumerate(file_lines)):
                dict_lines.append(func(json.loads(line)))
                if self.limit and i >= self.limit:
                    logger.warning(
                        f"Over dataset size limit {self.limit}, stop loading more."
                    )
                    break
            file_lines.close()
        return dict_lines


class CSVLoader(CategoryLoader[pd.DataFrame]):
    """
    Loads raw dataset in .csv format from a
    specific category.

    Parameters:
        category (str):
            The category of the dataset.
            E.g., "Video_Games", "Arts_Crafts_and_Sewing", etc.
        phase (str):
            The phase of the dataset.
            E.g., "raw", "grained".
        usage (str):
            The usage of the dataset.
            E.g., "meat", "train", "valid", "test".
    """

    def __init__(
        self, category: str, phase: str, usage: str = "", limit: Optional[int] = None
    ):
        super().__init__(
            category=category, ext="csv", phase=phase, usage=usage, limit=limit
        )

    def _load(self, file_path, func: Callable = lambda x: x) -> pd.DataFrame:
        df = pd.read_csv(
            file_path,
            sep=",",
            encoding="utf-8",
            nrows=self.limit if self.limit else None,
        )
        return df


def load_raw_data(category: str):
    """
    Load key contents of review and meta.
    Parameters:
        category (str):
            The category of the dataset.
            E.g., "Video_Games", "Books", etc.
    Returns:
        df_user_interact (np.DataFrame):
            A list of user interactions:
            Which user interacted with which
            item at which time.
        parentasin_title_map (List[dict]):
            A map from parent asin to title.
    """
    _parentasin_title_map = JsonlLoader(category=category, phase="raw", usage="meta")(
        func=lambda record: {
            "parent_asin": record["parent_asin"],
            "title": record["title"],
        }
    )

    parentasin_title_map = {
        record["parent_asin"]: record["title"] for record in _parentasin_title_map
    }

    title_itemid_map = {
        title: i for i, title in enumerate(parentasin_title_map.values())
    }

    df_user_interact = CSVLoader(
        category=category, phase="raw", usage="", limit=200000
    )()

    return df_user_interact, parentasin_title_map, title_itemid_map


def get_5core_ui_list(
    df_user_interact: pd.DataFrame, parentasin_title_map: dict, title_itemid_map: dict
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Using the raw dataset, build up a list of leave-one-out
    user interactions. The items are represented by:
        - parent_asin: The parent asin of the item.
        - item_title: The title of the item (further used).
        - item_id: The item ID.
    Parameters:
        df_user_interact (pd.DataFrame):
            A list of user interactions:
            Which user interacted with which
            item at which time.
        parentasin_title_map (List[dict]):
            A map from parent asin to title.
        title_itemid_map (dict):
            A map from title to item ID.
    Returns:
        train_list (List[dict]):
            A list of training records.
    """

    def __is_invalid(test_str: str) -> bool:
        return (pd.isna(test_str) or test_str == "" or test_str is None)
    
    # Sort the list by user_id, then by timestamp
    # To build up leave-one-out dataset
    df_user_interact.sort_values(["user_id", "timestamp"])

    # Search item title from paren_asin
    df_user_interact["item_title"] = df_user_interact["parent_asin"].map(
        parentasin_title_map
    )

    # Use sequencial item ID over strings
    df_user_interact["item_id"] = df_user_interact["item_title"].map(title_itemid_map)

    # List out key columns of each user
    key_concerns = ["parent_asin", "timestamp", "item_title", "item_id"]

    # Group users to list the above columns
    user_group = df_user_interact.groupby("user_id")
    user_data = user_group.agg(
        {key_concern: list for key_concern in key_concerns}
    ).to_dict("index")

    train_list: List[dict] = []
    valid_list: List[dict] = []
    test_list: List[dict] = []
    for user_id, interaction in user_data.items():

        (parentasin_list, timestamp_list, itemtitle_list, itemid_list) = [
            interaction[key_concern] for key_concern in key_concerns
        ]

        # User sequence is empty, drop the user.
        msg_user_seq_empty = (
            "[CSIT5210 Warning]: \n\nUser's interaction list is empty."
            f"\n\n user id: {user_id}, item title list: {itemtitle_list}"
            "\n\nSkip this user.\n\n")

        if len(itemtitle_list) == 0:
            logger.warning(msg_user_seq_empty)
            continue

        # Find all the invalid titles in advance.
        # Pad an invalid title at the end.
        invalid_title_indexes = [
            index for index, itemtitle in enumerate(itemtitle_list) 
            if __is_invalid(itemtitle)] + [len(itemtitle_list)]

        # Given all the invalid title indexes,
        # find all the valid title ranges.
        valid_title_ranges = (
            [(0, invalid_title_indexes[0])] + 
            [(
                invalid_title_indexes[i] + 1, invalid_title_indexes[i + 1])
                for i in range(0, len(invalid_title_indexes) - 1)
            ])

        # Pick out ranges that are too small.
        valid_title_ranges = [
            vt_range for vt_range in valid_title_ranges
            if vt_range[1] - vt_range[0] > 2
        ]

        if len(valid_title_ranges) <= 0:
            logger.warning(msg_user_seq_empty)
            continue

        # --- Convert user name to leave-one-out format ---

        # This user's leave-one-out record.
        this_user_record: List[dict] = []

        # Iterate all valid title ranges.
        for start, end in valid_title_ranges:

            # For a specific title range,
            # iterate all titles and generate leave-one-out records.
            for ptr_seq_end in range(start, end):
                new_record = {
                    "user_id": user_id,
                    "history_item_asins": parentasin_list[start:ptr_seq_end][-10:],
                    "new_item_asin": parentasin_list[ptr_seq_end],
                    "history_item_titles": itemtitle_list[start:ptr_seq_end][-10:],
                    "new_item_title": itemtitle_list[ptr_seq_end],
                    "history_item_ids": itemid_list[start:ptr_seq_end][-10:],
                    "new_item_id": itemid_list[ptr_seq_end],
                    "new_item_timestamp": timestamp_list[ptr_seq_end],
                }

                this_user_record.append(new_record)

        # Mostly, new records are given to training set.
        # For the last two records for each user, it will be given
        # to the validation set and test set respectively.

        if len(this_user_record) <= 2:
            # Only 2 records for this user,
            # meaning that there's only three items in the sequence.
            train_list.extend(this_user_record)
        else:
            train_list.extend(this_user_record[:-2])
            valid_list.append(this_user_record[-2])
            test_list.append(this_user_record[-1])

    df_train = pd.DataFrame(train_list)
    df_valid = pd.DataFrame(valid_list)
    df_test = pd.DataFrame(test_list)

    return df_train, df_valid, df_test


def grain_dataset(categories: List[str]):
    """
    Grain a list of given categories in the dataset.
    Parameters:
        categories (List[str]):
            A list of categories to be grained.
    """

    logger.info("[CSIT5210 Info]: \n\nGraining dataset started!\n\n")

    for category in categories:

        logger.info(f"[CSIT5210 Info]: \n\nGraining category {category} dataset...\n\n")

        base_path = f"data/grained/{category}"

        if not os.path.exists(base_path):
            logger.info(
                f"[CSIT5210 Info]: \n\nBase path {base_path} does not exist. "
                f"Creating...\n\n"
            )
            os.makedirs(base_path)
        elif bool(os.listdir(base_path)):
            logger.info(
                f"[CSIT5210 Info]: \n\nBase path {base_path} is not empty. "
                f"Category {category} grained! "
                f"Skipping...\n\n"
            )
            continue

        logger.info(f"[CSIT5210 Info]: \n\nLoading {category} raw 5-core data...\n\n")
        df_ui, pa_title_map, title_id_map = load_raw_data(category=category)

        logger.info(f"[CSIT5210 Info]: \n\nGraining {category} data...\n\n")
        df_train, df_valid, df_test = get_5core_ui_list(
            df_user_interact=df_ui,
            parentasin_title_map=pa_title_map,
            title_itemid_map=title_id_map,
        )

        logger.info(
            f"[CSIT5210 Info]: \n\nSaving {category} train, "
            f"valid and test .csv files...\n\n"
        )

        df_train.to_csv(os.path.join(base_path, f"train_{category}.csv"), index=False)
        df_valid.to_csv(os.path.join(base_path, f"valid_{category}.csv"), index=False)
        df_test.to_csv(os.path.join(base_path, f"test_{category}.csv"), index=False)


def mix_dataset(categories: List[str]):
    """
    Mix a list of given categories in the dataset.
    Parameters:
        categories (List[str]):
            A list of categories to be mixed.
    """

    logger.info("[CSIT5210 Info]: \n\nMixing Amazon dataset started!\n\n")

    mix_path = "data/grained/AmazonMix"
    if not os.path.exists(mix_path):
        logger.info(
            f"[CSIT5210 Info]: \n\nMix path {mix_path} does not exist. "
            f"Creating...\n\n"
        )
        os.makedirs(mix_path)
    elif bool(os.listdir(mix_path)):
        logger.info(
            f"[CSIT5210 Info]: \n\nMix path {mix_path} is not empty. "
            "AmazonMix dataset mixed! "
            "Stop mixing immediately.\n\n"
        )
        return

    all_train = pd.DataFrame()
    all_valid = pd.DataFrame()
    all_test = pd.DataFrame()

    stat = DefaultDict()

    for category in categories:
        logger.info(f"[CSIT5210 Info]: \n\nSampling category {category} dataset...\n\n")

        dataframe_train = CSVLoader(
            category=category, phase="grained", usage="train", limit=172747
        )()

        dataframe_valid = CSVLoader(
            category=category, phase="grained", usage="valid", limit=172747
        )()

        dataframe_test = CSVLoader(
            category=category, phase="grained", usage="test", limit=172747
        )()

        category_length = (
            len(dataframe_train) + len(dataframe_valid) + len(dataframe_test)
        )

        train_size = min(len(dataframe_train), math.floor(category_length * 0.8))
        valid_size = math.ceil(train_size * 0.125)
        test_size = math.ceil(train_size * 0.125)

        logger.info(
            f"[CSIT5210 Info]: \n\nCategory {category}:\n\n"
            f"train size: {train_size}, valid size: {valid_size} , test size: {test_size}\n\n"
        )

        stat[category] = {"train": train_size, "valid": valid_size, "test": test_size}

        dataframe_train = dataframe_train.sample(train_size, random_state=114)
        dataframe_valid = dataframe_train.sample(valid_size, random_state=114)
        dataframe_test = dataframe_train.sample(test_size, random_state=114)

        logger.info(
            f"[CSIT5210 Info]: \n\nAdding {category} to buffer dataframe...\n\n"
        )

        all_train = pd.concat([all_train, dataframe_train])
        all_valid = pd.concat([all_valid, dataframe_valid])
        all_test = pd.concat([all_test, dataframe_test])

    logger.info("[CSIT5210 Info]: \n\nSaving mixed dataset to .csv file...\n\n")

    all_train.to_csv(os.path.join(mix_path, "train_AmazonMix.csv"), index=False)
    all_valid.to_csv(os.path.join(mix_path, "valid_AmazonMix.csv"), index=False)
    all_test.to_csv(os.path.join(mix_path, "test_AmazonMix.csv"), index=False)

    logger.info(
        "[CSIT5210 Info]: \n\nMixed dataset saved!\n\n"
        f"Records: train {len(all_train)}, valid {len(all_valid)}, test {len(all_test)}."
        "Saving meta...."
    )

    meta_path = os.path.join(mix_path, "meta.json")
    with open(meta_path, mode="w", encoding="utf-8") as f:
        json.dump(stat, f, indent=4)
        f.close()

    logger.info(f"Meta saved at {meta_path}.")


def upload_dataset(categories: List[str]):
    """
    Upload Dataset to HuggingFace Hub.
    Parameters:
        categories (List[str]):
            A list of categories to be uploaded.
    """

    for category in categories:
        base_path = f"data/grained/{category}"

        df_train = pd.read_csv(os.path.join(base_path, f"train_{category}.csv"))
        df_valid = pd.read_csv(os.path.join(base_path, f"valid_{category}.csv"))
        df_test = pd.read_csv(os.path.join(base_path, f"test_{category}.csv"))

        (train_dataset, valid_dataset, test_dataset) = [
            Dataset.from_pandas(df) for df in [df_train, df_valid, df_test]
        ]

        dataset_dict = DatasetDict(
            {"train": train_dataset, "validation": valid_dataset, "test": test_dataset}
        )

        dataset_dict.push_to_hub(
            f"YzHuangYanzhen/Amazon-Reviews-2023-SR-L1O-{category}"
        )


if __name__ == "__main__":

    pretrain_categories = [
        "Video_Games",
        "Arts_Crafts_and_Sewing",
        "Movies_and_TV",
        "Home_and_Kitchen",
        "Electronics",
        "Tools_and_Home_Improvement",
    ]

    outofdomain_categories = ["Baby_Products", "Sports_and_Outdoors"]

    grain_dataset(categories=pretrain_categories + outofdomain_categories)
    mix_dataset(categories=pretrain_categories)
    upload_dataset(categories=pretrain_categories + outofdomain_categories)
