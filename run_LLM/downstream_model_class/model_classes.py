from typing import cast, Dict
from abc import abstractmethod

import torch
from torch import nn
from torch.nn.modules.loss import _Loss

from run_LLM.downstream_model_class.data_classes import (
    DownstreamModelArgs, 
    SASRecModelArgs, 
    # GRU4RecModelArgs,
    DownstreamTrainArgs)

from run_LLM.downstream_model_class.modules import TransformerEncoder


LOSS_FN_MAP: Dict[str, _Loss] = {
    "bce": nn.BCEWithLogitsLoss(),
    "ce": nn.CrossEntropyLoss()
}


class DownstreamModel(nn.Module):
    """
    Generic downstream model class.
    """

    def __init__(self, model_config: DownstreamModelArgs,
                 run_config: DownstreamTrainArgs):
        super(DownstreamModel, nn.Module).__init__()
        self.model_config = model_config
        self.run_config = run_config

    @abstractmethod
    def calculate_loss(self, batch):
        """
        Calculate loss.
        """
    
    @abstractmethod
    def predict(self, batch, n_return_sequences=1):
        """
        Perform prediction.
        """
    
    @abstractmethod
    def get_embeddings(self, items):
        """
        Obtain model embeddings.
        """


class MyEmbedding(nn.Module):

    def __init__(self, adapter: nn.Module, embedding: nn.Embedding):
        super().__init__()
        self.adapter = adapter
        self.embedding = embedding

    def forward(self, indices: torch.Tensor):
        return self.adapter(self.embedding(indices))
    
    @property
    def weight(self) -> torch.Tensor:
        return self.adapter(self.embedding.weight.data)


class SASRec(DownstreamModel):
    """
    SASRec model class.
    """
    
    def __init__(self, model_config: SASRecModelArgs, 
                 run_config: DownstreamTrainArgs,
                 pretrained_item_embeddings: torch.Tensor =None):
        super(SASRec, self).__init__(
            model_config=model_config,
            run_config=run_config)
        
        self.config = cast(SASRecModelArgs, self.config)

        assert self.config.adapter_dims[-1] == -1

        self.positional_embeddings = nn.Embedding(
            num_embeddings=self.run_config.max_seq_length,
            embedding_dim=self.model_config.hidden_size
        )

        self.item_embeddings = self.load_item_emb(
            pretrained_item_embeddings)

        self.emb_dropout = nn.Dropout(self.model_config.dropout)
        
        self.transformer_encoder = TransformerEncoder(self.config)
        
        self.loss_fn = nn.CrossEntropyLoss()


    def load_item_emb(self, pretrained_embs: torch.Tensor):
        """
        Attempt to load pretrained item embeddings.
        If no pretrained item embeddings provided,
        use a random embedding instead.
        """
        
        if pretrained_embs is None:
            item_emb = nn.Embedding(
                num_embeddings=self.run_config.item_num + 1,
                embedding_dim=self.model_config.hidden_size,
                padding_idx=0
            )
            nn.init.normal_(item_emb.weight, 0, 1)
            return item_emb
        
        assert pretrained_embs.shape[0] == self.run_config.item_num + 1
        
        # Reserved for further use.
        ext_emb = torch.randn(
            self.run_config.ext_token_num,
            pretrained_embs.shape[-1]).to(pretrained_embs.device)

        item_emb = nn.Embedding.from_pretrained(
            torch.cat([pretrained_embs, ext_emb]), padding_idx=0)
        
        # List of adapter (mlp) hidden sizes.
        adapter_hidden_sizes = [item_emb.embedding_dim] + self.config.adapter_dims
        adapter_hidden_sizes[-1] = self.config.hidden_size
        
        # Adapter mlp
        item_emb_adapter = nn.Sequential()

        item_emb_adapter.add_module(
            'lin_0',
            nn.Linear(adapter_hidden_sizes[0], 
                      adapter_hidden_sizes[1]))
        
        for i in range(1, len(adapter_hidden_sizes) - 1):
            item_emb_adapter.add_module(f'act_{i}', nn.ReLU())
            item_emb_adapter.add_module(f'lin_{i}',
                nn.Linear(adapter_hidden_sizes[i], 
                          adapter_hidden_sizes[i + 1]))
            
        # Init adapter mlp
        for name, param in item_emb_adapter.named_parameters():
            if "weight" in name:
                nn.init.xavier_normal_(param)
            elif "bias" in name:
                nn.init.constant_(param, 0)
        
        item_emb_ = MyEmbedding(adapter=item_emb_adapter, embedding=item_emb)
        return item_emb_


    def _get_representation(self, batch: dict):
        item_seqs = cast(torch.Tensor, batch["item_seqs"])
        input_embs = self.item_embeddings(item_seqs)
        input_embs += self.positional_embeddings(
            torch.arange(
                self.run_config.max_seq_length).to(input_embs.device)
        )

        seq = self.emb_dropout(input_embs)
        mask = torch.ne(item_seqs, 0).float().to(input_embs.device)
        mask = self.transformer_encoder.get_attn_mask(mask, bidirectional=False)
        
        seq_ = self.transformer_encoder(seq, attention_mask=mask)
        seq_ = cast(torch.Tensor, seq_)

        output = seq_[-1]  # Last transformer block's output.

        # Get the last item of all sequences in the batch.
        output = self.transformer_encoder.gather_batch_indices(
            output, batch["seq_lengths"] - 1)
        
        return output
    
    def forward(self, batch: dict):
        """
        Forward propagation and calculate loss.
        """
        
        reps = self._get_representation(batch)
        test_item_emb = self.item_embeddings.weight

        # Similarity
        logits = torch.matmul(reps, test_item_emb.transpose(0, 1))
        loss = self.loss_fn(logits, cast(torch.Tensor, batch["labels"]).view(-1))

        return {"loss": loss}
    
    def predict(self, batch: dict, n_return_sequences: int = 1):
        """
        Base on the given sequence, predict the next item.
        """
        reps = self._get_representation(batch).view(-1, self.model_config.hidden_size)
        test_item_emb = self.item_embeddings.weight
        logits = torch.matmul(reps, test_item_emb.transpose(0, 1))

        # Select
        s_from, s_to = self.run_config.select_pool
        scores = logits[:, s_from:s_to]
        preds = scores.topk(n_return_sequences, dim=-1).indices + s_from
        return preds
    

        

        
        