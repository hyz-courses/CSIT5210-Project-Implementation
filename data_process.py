"""
CSIT5210 - Data Mining and Knowledge Discovery
@author: HUANG, Yanzhen
@date: Oct. 10, 2025
@description: Data processing infrastructure.
"""

# Basics
import os
import ast
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
from utils.logs import bind_logger

logger = bind_logger(logger, log_path="logs/train_csft.log")

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
                f"The limit you inputted " f"is invalid. ({limit})"
            )
            logger.error(message)
            raise ValueError(message)

    @abstractmethod
    def _load(self, file_path, func: Callable = lambda x: x) -> T:
        pass

    @abstractmethod
    def _store(self, obj: T, file_path, func: Callable = lambda x: x) -> None:
        pass

    def load(self, func: Callable = lambda x: x) -> T:
        """
        Load the dataset file from a specific path.
        Parameters:
            func (Callable):
                The function to apply to the loaded object.
        Returns:
            T:
                The loaded object.
        """

        file_name = f"{self.category}.{self.ext}"

        if self.type != "":
            file_name = f"{self.type}_{file_name}"

        file_path = os.path.join("data", self.phase, self.category, file_name)

        if not file_path.endswith(f".{self.ext}"):
            message = (
                f"The file path you inputted "
                f"is not a valid .{self.ext} file. ({file_path})"
            )
            logger.error(message)
            raise ValueError(message)

        if not os.path.exists(file_path):
            message = f"The file {file_path} does not exist."
            logger.error(message)
            raise FileNotFoundError(message)

        return self._load(file_path, func)
    
    def store(self, obj: T, func: Callable = lambda x: x) -> None:
        """
        Store the processed data to a specific file format.
        Parameters:
            obj (T):
                The object to store.
            func (Callable):
                The function to apply to the object before storing.
        """

        base_path = f"data/{self.phase}/{self.category}"
        file_path = os.path.join(base_path, f"{self.type}_{self.category}.{self.ext}")
        
        if not os.path.exists(base_path):
            logger.info(
                f"Base path {base_path} does not exist. "
                f"Creating..."
            )
            os.makedirs(base_path)
        elif bool(os.path.exists(file_path)):
            logger.info(
                f"Base path {base_path} is not empty. "
                f"Category {self.category} grained! "
                f"Skipping..."
            )
            return
        
        file_path = os.path.join(base_path, f"{self.type}_{self.category}.{self.ext}")
        self._store(obj, file_path, func)

        logger.info(f"Category {self.category} file saved successfully to {file_path}.")


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

    def _store(self, obj: List[dict], file_path, func: Callable = lambda x: x) -> None:
        assert isinstance(obj, list)
        with open(file_path, mode='w', encoding='utf-8') as f:
            for record in func():
                f.write(json.dumps(record) + '\n')
            f.close()


class JsonLoader(CategoryLoader[dict]):
    """
    Loads raw dataset in .json format from a
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
            category=category, ext="json", phase=phase, usage=usage, limit=limit
        )

    def _load(self, file_path, func: Callable = lambda x: x) -> dict:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            f.close()
            return data

    def _store(self, obj: dict, file_path, func: Callable = lambda x: x) -> None:
        assert isinstance(obj, dict)
        with open(file_path, mode='w', encoding='utf-8') as f:
            obj_: dict = func(obj)
            json.dump(obj_, f, indent=4)
            f.close()


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

    def _store(self, obj: pd.DataFrame, file_path, func: Callable = lambda x: x) -> None:
        assert isinstance(obj, pd.DataFrame)
        obj.to_csv(file_path, index=False)
        

class TxtLoader(CategoryLoader[List[str]]):
    """
    Loads raw dataset in .txt format from a
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
            category=category, ext="txt", phase=phase, usage=usage, limit=limit
        )

    def _load(self, file_path, func: Callable = lambda x: x) -> List[str]:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = []
            for i, line in enumerate(f):
                lines.append(func(line))
                if self.limit and i >= self.limit:
                    logger.warning(
                        f"Over dataset size limit {self.limit}, "
                        f"stop loading more."
                    )
                    break
            f.close()
            return lines

    def _store(self, obj: List[str], file_path, func: Callable = lambda x: x) -> None:
        assert isinstance(obj, list)

        with open(file_path, 'w', encoding='utf-8') as f:
            for record in obj:
                f.write(record + '\n')
            f.close()


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
    _parentasin_title_map = JsonlLoader(category=category, phase="raw", usage="meta").load(
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
        category=category, phase="raw", usage="", limit=400000
    ).load()

    return df_user_interact, parentasin_title_map, title_itemid_map


def get_5core_ui_list(
    df_user_interact: pd.DataFrame, parentasin_title_map: dict, title_itemid_map: dict
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, set]:
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

    # Train, valid and test list of leave-one-out records.
    train_list: List[dict] = []
    valid_list: List[dict] = []
    test_list: List[dict] = []

    # Set of titles this user used.
    title_set = set()

    for user_id, interaction in user_data.items():

        (parentasin_list, timestamp_list, itemtitle_list, itemid_list) = [
            interaction[key_concern] for key_concern in key_concerns
        ]

        # User sequence is empty, drop the user.
        msg_user_seq_empty = (
            "User's interaction list is not valid to produce leave-one-out dataset."
            f"\n\nuser id: {user_id}, item title list: {itemtitle_list}"
            "\n\nSkip this user.")

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
        this_user_loo: List[dict] = []

        # This user's involved titles.
        this_user_title_set = set()

        # Iterate all valid title ranges.
        for start, end in valid_title_ranges:

            this_user_title_set.update(itemtitle_list[start:end])

            # For a specific title range,
            # iterate all titles and generate leave-one-out records.
            for ptr_seq_end in range(start + 1, end):
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

                this_user_loo.append(new_record)

        # Mostly, new records are given to training set.
        # For the last two records for each user, it will be given
        # to the validation set and test set respectively.

        if len(this_user_loo) <= 2:
            # Only 2 records for this user,
            # meaning that there's only three items in the sequence.
            train_list.extend(this_user_loo)
        else:
            train_list.extend(this_user_loo[:-2])
            valid_list.append(this_user_loo[-2])
            test_list.append(this_user_loo[-1])

        # Record this user's titles
        title_set.update(this_user_title_set)

    df_train = pd.DataFrame(train_list)
    df_valid = pd.DataFrame(valid_list)
    df_test = pd.DataFrame(test_list)

    return df_train, df_valid, df_test, title_set


def grain_dataset(categories: List[str]):
    """
    Grain a list of given categories in the dataset.
    Parameters:
        categories (List[str]):
            A list of categories to be grained.
    """

    logger.info("Graining dataset started!")

    for category in categories:

        logger.info(f"Graining category {category} dataset...")

        logger.info(f"Loading {category} raw 5-core data...")
        df_ui, pa_title_map, title_id_map = load_raw_data(category=category)

        logger.info(f"Graining {category} data...")
        df_train, df_valid, df_test, set_title = get_5core_ui_list(
            df_user_interact=df_ui,
            parentasin_title_map=pa_title_map,
            title_itemid_map=title_id_map,
        )

        logger.info(f"Saving {category} train, valid and test .csv files...")

        CSVLoader(category=category, phase='grained', usage='train').store(obj=df_train)
        CSVLoader(category=category, phase='grained', usage='valid').store(obj=df_valid)
        CSVLoader(category=category, phase='grained', usage='test').store(obj=df_test)
        TxtLoader(category=category, phase='grained', usage='titles').store(obj=list(set_title))


def mix_dataset(categories: List[str]):
    """
    Mix a list of given categories in the dataset.
    Parameters:
        categories (List[str]):
            A list of categories to be mixed.
    """

    logger.info("Mixing Amazon dataset started!")

    all_train = pd.DataFrame()
    all_valid = pd.DataFrame()
    all_test = pd.DataFrame()

    stat = DefaultDict()

    for category in categories:
        logger.info(f"Sampling category {category} dataset...")

        dataframe_train = CSVLoader(
            category=category, phase="grained", usage="train", limit=172747
        ).load()

        dataframe_valid = CSVLoader(
            category=category, phase="grained", usage="valid", limit=172747
        ).load()

        dataframe_test = CSVLoader(
            category=category, phase="grained", usage="test", limit=172747
        ).load()

        category_length = (
            len(dataframe_train) + len(dataframe_valid) + len(dataframe_test)
        )

        train_size = min(len(dataframe_train), math.floor(category_length * 0.8))
        valid_size = math.ceil(train_size * 0.125)
        test_size = math.ceil(train_size * 0.125)

        logger.info(
            f"Category {category}:\n"
            f"train size: {train_size}, valid size: {valid_size} , test size: {test_size}"
        )

        stat[category] = {"train": train_size, "valid": valid_size, "test": test_size}

        dataframe_train = dataframe_train.sample(train_size, random_state=114)
        dataframe_valid = dataframe_train.sample(valid_size, random_state=114)
        dataframe_test = dataframe_train.sample(test_size, random_state=114)

        logger.info(
            f"Adding {category} to buffer dataframe..."
        )

        all_train = pd.concat([all_train, dataframe_train])
        all_valid = pd.concat([all_valid, dataframe_valid])
        all_test = pd.concat([all_test, dataframe_test])

    logger.info("Saving mixed dataset to .csv file...")

    CSVLoader(category='AmazonMix', phase='grained', usage='train').store(obj=all_train)
    CSVLoader(category='AmazonMix', phase='grained', usage='valid').store(obj=all_valid)
    CSVLoader(category='AmazonMix', phase='grained', usage='test').store(obj=all_test)
    JsonLoader(category='AmazonMix', phase='grained', usage='meta').store(obj=stat)


    all_together = pd.concat([all_train, all_valid, all_test])
    mix_titleset = set()
    for _, row in all_together.iterrows():
        mix_titleset.add(row['new_item_title'])
        history_item_titles = ast.literal_eval(row["history_item_titles"])
        mix_titleset.update(history_item_titles)
    TxtLoader(category='AmazonMix', phase='grained', usage='titles').store(obj=list(mix_titleset))

    logger.info(
        "Mixed dataset saved!\n"
        f"Records: train {len(all_train)}, valid {len(all_valid)}, test {len(all_test)}."
        "Saving meta...."
    )


def upload_dataset(categories: List[str]):
    """
    Upload Dataset to HuggingFace Hub.
    Parameters:
        categories (List[str]):
            A list of categories to be uploaded.
    """

    for category in categories:
        df_train = CSVLoader(category=category, phase='grained', usage='train').load()
        df_valid = CSVLoader(category=category, phase='grained', usage='valid').load()
        df_test = CSVLoader(category=category, phase='grained', usage='test').load()

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
