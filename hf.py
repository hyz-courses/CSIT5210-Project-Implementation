import argparse
from huggingface_hub import HfApi, snapshot_download
from loguru import logger as _logger
from utils.logs import bind_logger

logger = bind_logger(_logger, log_path="logs/hf.log")

def push(local_dir: str, repo_id: str, repo_type: str):
    """
    Push a local file as a repository to Huggingface.
    """

    logger.info("Logging in...")

    api = HfApi()

    logger.info(
        f"Uploading folder "
        f"'{local_dir}' -> '{repo_id}' "
        f"({repo_type})")

    api.upload_large_folder(
        folder_path=local_dir,
        repo_id=repo_id,
        repo_type=repo_type)
    
    logger.info("Push completed!")


def pull(local_dir: str, repo_id: str, repo_type: str):
    """
    Pull a huggingface repo to local directory.
    """

    logger.info(
        f"Pulling repository "
        f"'{repo_id}' -> '{local_dir}' "
        f"({repo_type})")

    
    snapshot_download(
        repo_id=repo_id,
        repo_type=repo_type,
        local_dir=local_dir,
        local_dir_use_symlinks=False
    )

    logger.info("Pull completed!")


def main():
    parser = argparse.ArgumentParser()

    # Operation Type: Either push or pull

    group = parser.add_mutually_exclusive_group(required=True)

    group.add_argument(
        "--push", action="store_true", 
        help="Upload local folder to HF")
    
    group.add_argument(
        "--pull", action="store_true", 
        help="Download repo from HF to local")
    
    # Parameters
    
    parser.add_argument(
        "--local", type=str, 
        required=True, 
        help="Local folder to upload to hugginface."
    )

    parser.add_argument(
        "--repoid", type=str, 
        required=True, 
        help="Huggingface repo id."
    )

    parser.add_argument(
        "--repotype", type=str, 
        required=True, 
        choices=["model", "dataset"],
        help="Repository type: 'model' or 'dataset'."
    )

    args = parser.parse_args()

    if args.push:
        push(args.local, args.repoid, args.repotype)
    elif args.pull:
        pull(args.local, args.repoid, args.repotype)


if __name__ == "__main__":
    main()