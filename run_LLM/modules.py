from typing import List, Dict
from dataclasses import dataclass

import torch

@dataclass
class EvaluatorConfig:
    topks: List[int]
    eos_token: str


class Evaluator:
    """
    An evaluator class.
    """
    
    def __init__(self, evaluator_config: EvaluatorConfig):
        self.eos_token = evaluator_config.eos_token
        self.topks = evaluator_config.topks
        self.maxk = max(evaluator_config.topks)
    
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