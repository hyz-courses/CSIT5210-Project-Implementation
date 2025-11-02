import os
import json
from typing import (
    cast, get_args, List, Dict, 
    Union, Optional, Literal, Protocol)
from dataclasses import dataclass, asdict

# Tools
import numpy as np
import torch
from tqdm import trange

# LLM
from peft import PeftModel
from transformers import (
    AutoModel, AutoTokenizer,
    AutoConfig,
    PreTrainedModel, PreTrainedTokenizer,
    PretrainedConfig,
    LlamaConfig, MistralConfig,
    GemmaConfig, Qwen2Config,
)
from transformers.tokenization_utils_base import (
    BatchEncoding)
from transformers.modeling_outputs import BaseModelOutputWithPooling

from llm2vec.models import (
    MistralBiModel,
    LlamaBiModel,
    GemmaBiModel,
    Qwen2BiModel,
)

# Type definitions for linter type check.

LLM = Union[
    AutoModel,
    PreTrainedModel,
    MistralBiModel,
    LlamaBiModel,
    GemmaBiModel,
    Qwen2BiModel,
]

LLMConfig = Union[
    PretrainedConfig,
    AutoTokenizer,
    LlamaConfig,
    MistralConfig,
    GemmaConfig,
    Qwen2Config,
]

model_name_map: Dict[str, LLM] = {
    "Mistral": MistralBiModel,
    "Llama": LlamaBiModel,
    "Gemma": GemmaBiModel,
    "Qwen2": Qwen2BiModel,
}

token_styles = {
    "llama3": [
        "<|start_header_id|>user<|end_header_id|>\n\n{}<|eot_id|>", 
        "<|end_of_text|>"],
    "mistral/llama2": ["[INST] {} [/INST]", " </s>"],
    "gemma2": ["<bos><start_of_turn>user\n{}<end_of_turn>", "<eos>"],
    "qwen2": ["<|im_start|>user\n{}<|im_end|>", "<|endoftext|>"],
    "plain": "{}",
}

@dataclass
class EncoderArgs:
    """
    Encoder arguments.
    """
    
    pooling_mode: Optional[str] = None
    max_length: Optional[int] = None
    doc_max_length: Optional[int] = None
    skip_instruction: Optional[bool] = None

    def to_dict(self) -> dict:
        """
        Convert this class to dict.
        """
        return asdict(self)


ConvertOptions = Literal["numpy", "tensor"]

PoolingModes = Literal["mean", "weighted_mean", "eos_token", "bos_token"]

class PoolingFunction(Protocol):
    """
    A protocol class that defines how a pooling
    function should act.
    """
    def __call__(self,
                 features: BatchEncoding,
                 last_hidden_states: torch.FloatTensor,
                 **kwargs: object) -> torch.Tensor:
        ...


def pooling_function_factory(
        pooling_mode: str) -> PoolingFunction:
    """
    The factory function to create a pooling function.
    Parameters:
        pooling_mode (str): The pooling mode.
    Returns:
        PoolingFunction: The pooling function.    
    """
    if pooling_mode == "mean":
        def _mean(
                features: BatchEncoding, 
                last_hidden_states: torch.Tensor,
                **kwargs) -> torch.Tensor:
            """
            Take a mean over all the token embeddings. 
            """
            
            seq_len_list = features["attention_mask"].sum(dim=1)
            
            return torch.stack(
                [
                    last_hidden_states[i, -length:, :].mean(dim=0)
                    for i, length in enumerate(seq_len_list)
                ], dim=0
            )
        return _mean
    
    if pooling_mode == "weighted_mean":
        def _weighted_mean(
                features: BatchEncoding, 
                last_hidden_states: torch.Tensor,
                **kwargs) -> torch.Tensor:
            
            seq_len_list = features["attention_mask"].sum(dim=1)
            
            batch_size, length, _ = last_hidden_states.shape

            token_weights = torch.zeros(
                batch_size, length, 
                device=last_hidden_states.device)

            for i, seq_len in enumerate(seq_len_list):
                if seq_len <= 0:
                    continue

                token_weights[i, -seq_len:] = torch.arange(seq_len) + 1
                token_weights[i] /= torch.clamp(
                    token_weights[i].sum(), min=1e-9)

            return torch.sum(
                last_hidden_states * token_weights.unsqueeze(-1), dim=1)
        
        return _weighted_mean
    
    if pooling_mode in ["eos_token", "last_token"]:
        
        def _eos_token(
                features: BatchEncoding, # pylint: disable=unused-argument
                last_hidden_states: torch.Tensor,
                **kwargs) -> torch.Tensor:
            return last_hidden_states[:, -1]
        
        return _eos_token
    
    if pooling_mode == "bos_token":
        def _bos_token(
                features: BatchEncoding, 
                last_hidden_states: torch.Tensor,
                **kwargs) -> torch.Tensor:
            
            bos_token_id = kwargs.get("bos_token_id", None)

            assert (
                bos_token_id is not None 
                and isinstance(bos_token_id, int))

            return last_hidden_states[
                features["input_ids"] == bos_token_id
            ]
        
        return _bos_token
    
    raise NotImplementedError(
        f"Your required pooling mode {pooling_mode} is unknown.")


class LLM2Vec(torch.nn.Module):
    """
    A generic LLM2Vec encoder model
    that can load multiple types of LLMs.
    """

    def __init__(
        self, model: LLM,
        tokenizer: AutoTokenizer,
        pooling_mode: str = "mean",
        max_length: int = 512,
        doc_max_length: int = 400,
        skip_instruction: bool = False
    ):
        
        super().__init__()
        self.model: LLM = model
        self.tokenizer: PreTrainedTokenizer = tokenizer
        self.config: PretrainedConfig = model.config

        # Encoder Args
        self.pooling_mode = pooling_mode
        self.skip_instruction = skip_instruction
        self.max_length = max_length
        self.doc_max_length = doc_max_length

        self.encoder_args = EncoderArgs(
            pooling_mode=pooling_mode,
            max_length=max_length,
            doc_max_length=doc_max_length,
            skip_instruction=skip_instruction
        )

        # Others
        self.pooling_function = pooling_function_factory(pooling_mode)

        assert self.tokenizer.padding_side == "left", "Padding side should be left."

    def _txtlen(self, text) -> int:
        """
        The length of a token.
        """

        if not hasattr(text, "__len__"):
            return 1

        if isinstance(text, dict):
            return len(text.keys())
            
        if (
            isinstance(text, str)
            or ( # A single embedding
                isinstance(text, list) 
                and all([isinstance(x, int) for x in text]))):
            return len(text)

        # TODO: Stat model output and seek for a better solution.
        return sum([len(t) for t in text])
        

    def _seasoning(self, text: str) -> str:
        """
        Season an input text into the target model's dialect
        by adding styled tokens.

        Parameters
            text: str
                Input text to season.
        
        Returns
            str: Seasoned text.
        """
        

        if isinstance(self.model, MistralBiModel):
            token_style = "mistral/llama2"
        elif isinstance(self.model, LlamaBiModel):
            if "Llama-2" in getattr(self.model.config, "_name_or_path", ""):
                token_style = "mistral/llama2"
            else:
                token_style = "llama3"
        elif isinstance(self.model, GemmaBiModel):
            token_style = "gemma2"
        elif isinstance(self.model, Qwen2BiModel):
            token_style = "qwen2"
        else:
            token_style = "plain"
        
        seasoned_text = token_styles[token_style][0].format(text.strip())

        if self.pooling_mode == "eos_token":
            seasoned_text += token_styles[token_style][1]
        
        return seasoned_text
    

    def _splice_instruction(self, instruction: str, text: str) -> str:
        """
        Splice instruction and text into a single,
        formatted expression. Precisely control token
        number to fit max doc length while not over-cutting
        token numbers.

        Parameters
            instruction: str
                Instruction to splice.
            text: str
                Text.
        Returns
            str: The formatted expression.
        """
        formatter = "{}ε٩(๑>₃<)۶з{}"

        def tokenize(text: str) -> BatchEncoding:
            """
            Tokenize a text into a list of tokens.
            {
                "input_ids": tensor([[101, 2023, 2003, 1037, 102]]),
                "attention_mask": tensor([[1, 1, 1, 1, 1]])
            }
            """
            
            return self.tokenizer(
                text, 
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_length,
                add_special_tokens=False,
            )
        
        text_tk = tokenize(text)
        len_text_tk = len(text_tk["input_ids"][0]) # num. of tokens

        if len_text_tk < self.doc_max_length:
            return formatter.format(instruction, text)

        # Binary search for the best text length.
        words = text.split()    # List of words (human's perspective)
        left, right = 0, len(words)
        best_text = text

        while left < right:
            mid = (left + right) // 2
            candidate_text = " ".join(words[:mid])
            candidate_tk = tokenize(candidate_text)
            len_candidate_tk = len(candidate_tk["input_ids"][0])

            if len_candidate_tk <= self.doc_max_length:
                best_text = candidate_text
                right = mid - 1
            else:
                left = mid + 1
        
        return formatter.format(instruction, best_text)


    def _tokenize(self, texts: List[str])-> BatchEncoding:
        """"
        Tokenize a list of formatted instruction-text strings.
        Each string's instruction and text are separated by a
        delimiter of `ε٩(๑>₃<)۶з`. 
        
        For each instruction-text string, obtain: 1. The text
        only embedding, and 2. The combined embedding (without
        delimiter). Mask out each sentence's combind emedding's
        answer part. Outputs the masked combined embeddings.

        Example
        >>> texts = [
        >>> "Read ε٩(๑>₃<)۶зthe book",
        >>> "Search ε٩(๑>₃<)۶зon the world wide web"
        >>> ]
        >>> text_list = ["the book", "on the world wide web"]
        >>> instruction_text_list = ["Read the book", "Search on the world wide web"]
        >>> embed_mask = [
        >>> [0, 1, 1], 
        >>> [0, 1, 1, 1, 1, 1]
        >>> ]
        
        Parameters
            texts: List[str]
                List of formatted instruction-text strings.
        
        Returns
            BatchEncoding: The tokenized instruction-text strings.
        """
        
        delimiter = "ε٩(๑>₃<)۶з"
        text_list: List[str] = []
        instruction_text_list: List[str] = []

        for text in texts:
            it_pair = text.split(delimiter)
            _, text = it_pair

            if len(text) <= 1:
                text = ""
            
            text_list.append(text)
            instruction_text_list.append("".join(it_pair))

        # Instruction-text embedding.
        it_emb: BatchEncoding = self.tokenizer(
            instruction_text_list,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length
        )

        embed_mask = None

        for i, text in enumerate(text_list):
            # Text-only embedding.
            t_emb: BatchEncoding = self.tokenizer(
                [text],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_length,
                add_special_tokens=False,
            )

            # Mask out answers in instruction-text strings.
            e_m = torch.zeros_like(it_emb["attention_mask"][i])
            if len(t_emb["input_ids"][0]) > 0:
                e_m[-len(t_emb["input_ids"][0]) :] = torch.ones(
                    len(t_emb["input_ids"][0]))
            
            if embed_mask is None:
                embed_mask = e_m.unsqueeze(0)
            else:
                embed_mask = torch.cat(
                    [embed_mask, e_m.unsqueeze(0)], dim=0)
        
            assert e_m.shape == cast(
                torch.Tensor, it_emb["attention_mask"]).shape

        it_emb["embed_mask"] = embed_mask

        return it_emb


    @classmethod
    def from_pretrained(
        cls, base_model_name_or_path: str,
        peft_model_name_or_path: Optional[str] = None,
        merge_peft: bool = False,
        bidirectional: bool = True,
        **kwargs,
    ):
        """
        Load pretrained weights.
        """

        # Encoder args
        encoder_args = EncoderArgs(**kwargs)
        
        # Tokenizer 
        tokenizer = AutoTokenizer.from_pretrained(
            base_model_name_or_path
        )
        tokenizer = cast(PreTrainedTokenizer, tokenizer)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"

        # Base Model Config -> Config Name -> Model Class
        #        `--------------------------------^
        basemodel_autoconfig = AutoConfig.from_pretrained(
            base_model_name_or_path)
        basemodel_autoconfig = cast(LLMConfig, basemodel_autoconfig)

        if not bidirectional:
            model_class = AutoModel
        else:
            model_class: LLM = model_name_map[
                basemodel_autoconfig.__class__.__name__.replace("Config", "")
            ]
        
        model = model_class.from_pretrained(
            base_model_name_or_path, 
            **encoder_args.to_dict()
        )

        # Sync the model's stored path with the actual path.
        # Latest: base_model_name_or_path
        sync_config_file: str = os.path.join(
            base_model_name_or_path, "config.json"
        )

        if os.path.exists(sync_config_file):
            with open(
                sync_config_file, 
                "r", encoding="utf-8") as f:
                sync_config_dict = json.load(f)
                f.close()
            sync_config = PretrainedConfig.from_dict(sync_config_dict)
            sync_config_name_or_path = getattr(sync_config, "_name_or_path", None)
            model_config_name_or_path = getattr(model.config, "_name_or_path", None)
            if (sync_config_name_or_path is not None
                and sync_config_name_or_path != model_config_name_or_path):
                setattr(model.config, "_name_or_path", sync_config_name_or_path)

        # Load PEFT model
        if hasattr(model, "peft_config"):
            # config.json and adapter
            # are in the same dir
            model = PeftModel.from_pretrained(model, base_model_name_or_path)
            model = model.merge_and_unload()
        elif peft_model_name_or_path is not None:
            # otherwise, manually provide
            # adapter path
            model = PeftModel.from_pretrained(model, peft_model_name_or_path)
            if merge_peft:
                model = model.merge_and_unload()

        # Update model config with llm2vec_config.json
        config_file: str = os.path.join(
            (peft_model_name_or_path or base_model_name_or_path),
            "llm2vec_config.json")
        
        new_config = {}
        if os.path.exists(config_file):
            with open(config_file, "r", encoding="utf-8") as f:
                new_config = json.load(f)
                f.close()

        new_config.update(encoder_args.to_dict())
        
        return cls(model=model, tokenizer=tokenizer, **new_config)


    def forward(self, sentence_feature: BatchEncoding) -> torch.Tensor:
        """
        Overwrite forward() of torch.nn.Module.
        Given a sentence feature, which is a BatchEncoding
        object containing the embedding list, attention 
        mask and embedding mask, return the pooled embedding
        of the entire sentence.

        Parameters
            sentence_feature: BatchEncoding
                A BatchEncoding object containing the sentence's 
                embedding list, attention mask and embedding mask.
        
        Returns
            torch.Tensor:
                The sentence embedding.
        """
        
        reps: BaseModelOutputWithPooling = self.model(**{
            k:v for k, v in sentence_feature.items()
            if k != "embed_mask"
        })

        if self.skip_instruction:
            sentence_feature["attention_mask"] = sentence_feature["embed_mask"]

        return self.pooling_function(
            sentence_feature, 
            reps.last_hidden_state,
            bos_token_id=self.tokenizer.bos_token_id)

    def encode(
            self, sentences: Union[str, list],
            batch_size: int = 32,
            convert_to: str = ConvertOptions,
            device: Optional[str] = None,
        ) -> Union[torch.Tensor, np.ndarray]:

        """
        Encode sentences to embeddings. Each sentence
        corresponds to a single embedding.
        """
        
        # Convert options restricted.
        if (not isinstance(convert_to, str) 
            or convert_to not in get_args(ConvertOptions)):
            raise ValueError(
                f"Invalid convert option {convert_to}.")
        
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        # Unify param. type
        if isinstance(sentences, str):
            sentences = [["", sentences]]
        elif (isinstance(sentences, list) 
              and isinstance(sentences[0], str)):
            if isinstance(sentences[-1], int):
                sentences = [[sentences]]
            else:
                sentences = [["", s] for s in sentences]
        else:
            raise ValueError("Invalid input type.")

        # Splice instruction and text.
        sentences: List[str] = [self._splice_instruction(
            instruction=s[0], text=s[1]
        ) for s in sentences]

        # Sort sentences by length DESC.
        sent_idx_desc = np.argsort([-self._txtlen(s) for s in sentences])
        sentences: List[str] = [sentences[i] for i in sent_idx_desc]

        self.eval()
        self.to(device)

        embedding_list: List[torch.Tensor] = []

        for start in trange(0, len(sentences), batch_size):
            sentences_batch = sentences[start:start+batch_size]

            seasoned_batch: List[str] = [
                self._seasoning(s) for s in sentences_batch]

            # Obtain features of this sentence.
            features: BatchEncoding = self._tokenize(seasoned_batch)
            
            for key in features:
                if isinstance(features[key], torch.Tensor):
                    features[key] = features[key].to(device)

            with torch.no_grad():
                # Sentence embedding
                emb = self.forward(features)
                emb = emb.detach()
                emb = emb.cpu()
                embedding_list.append(emb)
        
        # Collect sentence embeddings
        embedding_list = torch.cat(embedding_list, dim=0)
        embedding_list = embedding_list[np.argsort(sent_idx_desc)]
        embedding_list = embedding_list.to(torch.float32)

        if convert_to == "numpy":
            embedding_list = np.asarray([
                emb.numpy() for emb in embedding_list])

        return embedding_list


    def save(self, output_dir: str, 
             merge_before_save: bool = False,
             save_config: bool = True):
        """
        Save the model and encoder config.
        Parameters
            output_dir: str
                The directory to save the model.
            merge_before_save: bool
                Whether to merge the adapter before saving.
            save_config: bool
                Whether to save the encoder config.
        """
        
        if isinstance(self.model, PeftModel) and merge_before_save:
            self.model = self.model.merge_and_unload()
            if hasattr(self.model, "_hf_peft_config_loaded"):
                setattr(self.model, "_hf_peft_config_loaded", False)

        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)

        llm2vec_config = self.encoder_args.to_dict()

        if save_config:
            llm2vec_config_path = os.path.join(
                output_dir, "llm2vec_config.json")
            os.makedirs(output_dir, exist_ok=True)
            with open(llm2vec_config_path, "w", encoding="utf-8") as f:
                json.dump(llm2vec_config, f, indent=4)
                f.close()

# class LLM2VecEncoder:
#     def __init__(
#             self, model_path: str, 
#             peft_model_name_or_path: str, 
#             bidirectional: bool):
        
#         config = {
#             "base_model_name_or_path": model_path,
#             "peft_model_name_or_path": peft_model_name_or_path,
#             "device_map": "cuda" if torch.cuda.is_available() else "cpu",
#             "torch_dtype": torch.bfloat16,
#             "use_auth_token": os.environ.get("HUGGINGFACE_HUB_TOKEN"),
#             "enable_bidirectional": bidirectional
#         }

#         if bidirectional:
#             config["pooling_mode"]="eos_token"
        
