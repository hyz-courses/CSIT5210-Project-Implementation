import os
import math
import hashlib
from copy import copy
from datetime import datetime
from typing import cast, Dict
from collections import defaultdict, OrderedDict
from dataclasses import asdict

import torch
import wandb
from tqdm import tqdm
from torch.optim import AdamW
from torch.utils.data import DataLoader
from accelerate import Accelerator
from loguru import logger as _logger


from data_process import NpyLoader, JsonLoader
from train_LLM.modules import TrainSuite
from run_LLM.datasest import IDRecDatasets
from run_LLM.downstream_model_class.data_classes import (
    SASRecModelArgs,
    GRU4RecModelArgs,
    DownstreamTrainArgs)
from run_LLM.downstream_model_class.model_classes import (
    DownstreamModel, 
    SASRec, GRU4Rec)
from utils.logs import bind_logger
from utils.reproduce import freeze_random

THIS_FILE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT_DIR = os.path.join(THIS_FILE_DIR, "..")

logger = bind_logger(_logger,
                     log_path=os.path.join(
                        PROJECT_ROOT_DIR,
                        "logs", "downstream.log"
                     ))


class Evaluator:
    """
    An evaluator class.
    """
    
    def __init__(self, evaluator_config: DownstreamTrainArgs):
        self.eos_token = evaluator_config.eos_token
        self.topks = evaluator_config.topk
        self.maxk = max(evaluator_config.topk)
    
    def _calc_pos_index(
            self, preds: torch.Tensor, 
            labels: torch.Tensor) -> torch.Tensor:
        
        """
        For a batch of users, do the following:

        Given a users' top-maxk prediction
        candidates (maxk is the max topk number),
        and the label, check which index the label
        is in the user's prediction.

        Each user corresponds to a row boolean
        vector, where only the index of the label
        is true. If label is not in the candidate,
        the entire boolean vector will be false.

        Parameters:

            preds (Tensor):
                The batch of user top-maxk prediction 
                candidates. dim0 = #. users, dim1 = maxk.
            
            labels (Tensor):
                A list of user ground-truths.
        
        Returns:
            BoolTensor:
                A boolean map same as preds.

        Example:
        >>> preds = [
        >>> [11, 45, 14, 19, 81],
        >>> [81, 88, 75, 57, 16],
        >>> [34, 25, 1, 22, 56]
        >>> ]
        >>> labels = [14, 57, 89]
        >>> pos_index = [
        >>> [False, False, True, False, False],
        >>> [False, False, False, True, False],
        >>> [False, False, False, False, False]
        >>> ]
        """

        # Preds: dim0: all users, dim1: topk
        preds = preds.detach().cpu()
        labels = labels.detach().cpu()

        assert preds.shape[1] == self.maxk

        # Pos: dim0: all users
        # dim1: which candidate?
        pos_index = torch.zeros(
            (preds.shape[0], self.maxk),
            dtype=torch.bool)
        
        for i in range(preds.shape[0]):
            # all users
            label = labels[i].tolist()

            # Cast label to end token
            if self.eos_token in [label]:
                eos_pos = label.index(self.eos_token)
                label = label[:eos_pos]

            for j in range(self.maxk):
                # i-th user's, j-th candidate
                pred = preds[i, j].tolist()
                if pred == label:
                    pos_index[i, j] = True
                    break
        
        return pos_index
    

    def _calc_recall_at_k(
            self, pos_index: torch.Tensor, 
            k: int) -> torch.Tensor:
        """
        For each user specifically, how many correct
        predictions in the first k candidates.

        Parameters:
            pos_index (Tensor):
                The boolean map of user prediction.
                Shape: #. users x maxk
            k (int):
                Clip at which k, shouldn't be larger
                than maxk.
        Return:
            Tensor: The recall@k.
        """
        return pos_index[:, :k].sum(dim=1).cpu().float()
    

    def _calc_ndcg_at_k(
            self, pos_index: torch.Tensor, 
            k: int) -> torch.Tensor:
        """
        NDCG@k, i.e., Normalized Discount Cumulative Gain.
        Calculate NDCG@k for a batch of users.

        Parameters:
            pos_index (Tensor):
                The boolean map of user prediction.
                Shape: #. users x maxk
            k (int):
                Clip at which k, shouldn't be larger
                than maxk.
        Return:
            Tensor: The NDCG@k.
        """
        # Range from 1 to maxk for calculate DCG.
        ranks = torch.arange(
            1, pos_index.shape[-1] + 1
            ).to(pos_index.device)
        dcg = 1.0 / torch.log2(ranks + 1)
        dcg = torch.where(
            condition=pos_index, input=dcg, 
            other=torch.tensor(
                0.0, dtype=dcg.dtype, 
                device=dcg.device))
        
        return dcg[:, :k].sum(dim=1).cpu().float()
    

    def calc(
            self, preds: torch.Tensor, 
            labels: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Given a batch of user prediction top-maxk lists,
        and the correct lables, obtain recall@k and ndcg@k
        for all wanted k.

        Parameters:

            preds (Tensor):
                The batch of user top-maxk prediction 
                candidates. dim0 = #. users, dim1 = maxk.
            
            labels (Tensor):
                A list of user ground-truths.

        Returns:
            dict:
                A result dictionary contains recall@k and
                ndcg@k for all defined k.

        """
        pos_index = self._calc_pos_index(preds=preds, labels=labels)
        results = {}
        for k in self.topks:
            recall_at_k = self._calc_recall_at_k(pos_index=pos_index, k=k)
            ndcg_at_k = self._calc_ndcg_at_k(pos_index=pos_index, k=k)
            results.update({
                f"recall@{k}": recall_at_k,
                f"ndcg@{k}": ndcg_at_k
            })

        return results


class DownstreamTrainSuite(TrainSuite):
    """
    An inheritence of TrainSuite, for downstream tasks.
    """

    def __init__(self, 
                 model: DownstreamModel, 
                 accelerator: Accelerator,
                 run_config: DownstreamTrainArgs,
                 train_loader: DataLoader,
                 valid_loader: DataLoader):
        super().__init__(_logger=logger)
        self.run_config = run_config
        self.model = model

        self.train_loader = train_loader
        self.valid_loader = valid_loader

        self.accelerator = accelerator
        self.evaluator = Evaluator(run_config)
        self.save_model_ckpt = os.path.join(
            PROJECT_ROOT_DIR, self.run_config.ckpt_dir,
            self._generate_ckpt_filename()
        )
        os.makedirs(os.path.dirname(self.save_model_ckpt), exist_ok=True)

        self.best_metric = 0
        self.best_epoch = 0
        self.count = 0

    def _get_model_copy(self):
        return copy(self.model)


    def _generate_ckpt_filename(self):
        """
        Base on the time, run id and config, generate
        a unique checkpoint filename.
        """
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        md5 = hashlib.md5(str(asdict(self.run_config)).encode(encoding="utf-8")).hexdigest()[:6]
        return f"{self.run_config.run_id}-{now}-{md5}.pth"


    def _evaluate(self, valid_loader: DataLoader):
        """
        Perform one evaluation on the validation set.
        """

        self.model.eval()

        _summary = defaultdict(list)

        for batch in tqdm(
            valid_loader,
            total=len(valid_loader),
            desc="DS-Evaluation"
        ):
            with torch.no_grad():
                preds = self.model.predict(
                    batch, n_return_sequences=self.evaluator.maxk)
                
                metrics = self.evaluator.calc(
                    cast(torch.Tensor, preds), 
                    cast(torch.Tensor, batch["labels"]))
                
                for k, v in metrics.items():
                    _summary[k].append(v)

        summary = OrderedDict()
        for k, val_list in _summary.items():
            mean = torch.cat(val_list).mean().item()
            summary[k] = mean
        
        return summary

    def train(self):
        """
        Perform training along with validation.
        Early stop when overfit.
        """
        optimizer = AdamW(
            self.model.parameters(),
            lr=self.run_config.lr,
            weight_decay=self.run_config.weight_decay
        )

        (
            self.model, optimizer,
            train_loader, valid_loader
        ) = self.accelerator.prepare(
            self.model, optimizer, 
            self.train_loader, self.valid_loader)

        self.model = cast(DownstreamModel, self.model)
        optimizer = cast(AdamW, optimizer)
        train_loader = cast(DataLoader, train_loader)
        valid_loader = cast(DataLoader, valid_loader)

        self.accelerator.init_trackers(
            project_name="CSIT5210-Impl-G1",
            config=asdict(self.run_config)
        )

        n_epochs: int = math.ceil(
            self.run_config.epochs / self.accelerator.num_processes)

        best_epoch = -1
        best_val_score = -1

        for epoch in range(n_epochs):
            self.model.train()
            train_loss = 0.0

            for batch in tqdm(
                train_loader,
                total=len(train_loader),
                desc=f"DS-Train [Epoch {epoch + 1}/{n_epochs}]"
            ):
                optimizer.zero_grad()
                outputs = self.model(batch)
                loss = cast(torch.Tensor, outputs["loss"])
                self.accelerator.backward(loss)
                optimizer.step()
                train_loss += loss.item()

            self.accelerator.log({
                "train_loss": train_loss / len(train_loader)
                }, step=epoch + 1)
            
            if (epoch + 1) % self.run_config.eval_interval != 0:
                continue

            metrics_summary = self._evaluate(valid_loader)

            if self.accelerator.is_main_process:
                for key in metrics_summary:
                    self.accelerator.log({
                        f"validation/{key}": metrics_summary[key]
                    }, step=epoch + 1)

            # Use recall@20 as main metric
            recall_at_maxk = metrics_summary[f"recall@{self.evaluator.maxk}"]

            if recall_at_maxk > best_val_score:
                best_val_score = recall_at_maxk
                best_epoch = epoch + 1

                if self.accelerator.is_main_process:
                    # torch.save(self.model.state_dict(), self.save_model_ckpt)
                    self.save()
                    logger.info(
                        f"Checkpoint saved to {self.save_model_ckpt} "
                        f"at epoch {epoch + 1}.")

            if epoch + 1 - best_epoch >= self.run_config.patience:
                logger.info(
                    f"Stop early at epoch {epoch + 1}.\n"
                    f"Best epoch: {best_epoch}; "
                    f"Best validation score: {best_val_score}")
                break


    def evaluate(self, loader: DataLoader):
        """
        API that exposes internal method.
        """
        return self._evaluate(loader)

    def end(self):
        """
        End training.
        """
        self.accelerator.end_training()

    def save(self):
        """
        Save model.
        """
        torch.save(self.model.state_dict(), self.save_model_ckpt)


class Main:
    """
    Main running instance.
    """
    def __init__(
            self, 
            category: str,
            run_config: DownstreamTrainArgs, 
            which_downstream_model: str,):
        
        logger.info(f"Evaluating Category: {category} with {which_downstream_model}.")
        
        self.run_config = run_config
        self.category = category
        self.which_downstream_model = which_downstream_model
        run_config.run_id = which_downstream_model
        
        freeze_random(run_config.rand_seed)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        wandb.init(
            project=f"CSIT5210-Downstream-{which_downstream_model}",
            name=run_config.run_id)
        
        self.accelerator = Accelerator(log_with="wandb")

        (
            self.train_dataset, self.valid_dataset, 
            self.test_dataset, total_item_num, 
            select_pool) = IDRecDatasets(category).get_datasets()
        
        self.run_config.select_pool = select_pool
        self.run_config.eos_token = total_item_num + 1
        self.run_config.item_num = total_item_num

        # Load embedding from .npy file
        if self.run_config.use_pretrained_embedding:
            _pretrained_item_embeddings = NpyLoader(
                category=category,
                phase="downstream",
                usage="emb",
                project_root=PROJECT_ROOT_DIR
            ).load()
            pretrained_item_embeddings = torch.tensor(
                _pretrained_item_embeddings, 
                dtype=torch.float32).to(self.device)
        else:
            pretrained_item_embeddings = None
        
        with self.accelerator.main_process_first():
            
            if "SASRec" in which_downstream_model:
                model_args = SASRecModelArgs()
                self.model = SASRec(model_args, run_config, pretrained_item_embeddings)
            elif "GRU4Rec" in which_downstream_model:
                model_args = GRU4RecModelArgs()
                self.model = GRU4Rec(model_args, run_config, pretrained_item_embeddings)
            else:
                raise ValueError(f"Unknown downstream model type {which_downstream_model}.")
        
        self.train_suite = DownstreamTrainSuite(
            model=self.model, 
            accelerator=self.accelerator, 
            run_config=self.run_config,
            train_loader = DataLoader(
                dataset=self.train_dataset, 
                batch_size=self.run_config.train_batch_size,
                shuffle=True),
            valid_loader = DataLoader(
                dataset=self.valid_dataset, 
                batch_size=self.run_config.eval_batch_size,
                shuffle=False)
        )

    def main(self):

        if JsonLoader(
            category=self.category,
            phase="result",
            usage=self.which_downstream_model,
            limit=None,
            project_root=PROJECT_ROOT_DIR
        ).exist():
            logger.info(f"Downstream eval results for category {self.category} exists, skip training.")
            return

        # Train downstream
        self.train_suite.train()
        self.accelerator.wait_for_everyone()
        self.model = self.accelerator.unwrap_model(self.model)

        # Test downstream
        self.model = cast(SASRec, self.model)
        self.model.load_state_dict(torch.load(self.train_suite.save_model_ckpt))
        self.model, test_dataloader = self.accelerator.prepare(
            self.model,
            DataLoader(self.test_dataset, 
                       batch_size=self.run_config.eval_batch_size,
                       shuffle=False)
        )

        self.model = cast(SASRec, self.model)
        test_dataloader = cast(DataLoader, test_dataloader)

        test_results = self.train_suite.evaluate(test_dataloader)
        if self.accelerator.is_main_process:
            for k, v in test_results.items():
                self.accelerator.log({f"test/{k}": v})

        # breakpoint()

        JsonLoader(
            category=self.category,
            phase="result",
            usage=self.which_downstream_model,
            project_root=PROJECT_ROOT_DIR
        ).store(obj=test_results)

        self.train_suite.end()
        return test_results, self.run_config
