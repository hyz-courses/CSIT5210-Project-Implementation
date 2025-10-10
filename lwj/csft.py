import os
os.environ["NCCL_P2P_DISABLE"] = "1"  # 禁用 NVLink
os.environ["NCCL_IB_DISABLE"] = "1"   # 禁用 InfiniBand，如果适用
os.environ["NCCL_NET_GDR_LEVEL"] = "0"  # 禁用 GDR（GPU 直连）
from dataset import PromptDataset
from datasets import Dataset as HFDataset
import argparse
import fire
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, EarlyStoppingCallback
def parse_args():
    parser = argparse.ArgumentParser(description="CSFT Training Script")

    parser.add_argument("--base_model", type=str, required=True, help="预训练模型路径")
    parser.add_argument("--train_data_path", type=str, required=True, help="训练数据文件路径")
    parser.add_argument("--eval_data_path", type=str, required=True, help="验证数据文件路径")
    parser.add_argument("--output_dir", type=str, required=True, help="模型输出目录")
    parser.add_argument("--wandb_run_name", type=str, required=True, help="wandb 跟踪名称")

    return parser.parse_args()

def train(
        #file path:model, data, output
        base_model_path: str = "",
        train_data_path: str = "",
        eval_data_path: str = "",
        output_dir: str ="./output/Qwen2-0.5B-CSFT-",

        #train setting
        batch_size: int = 128,
        micro_batch_size: int = 4,
        num_epochs: int = 10,
        learning_rate: float = 3e-4,
        cutoff_len: int = 1024,
        #llm setting
        group_by_length: bool = False,  # faster, but produces an odd training loss curve
        #wandb setting
        wandb_name: str = "Qwen2-0.5B-CSFT-AmazonMix-6",
        #others
        K=0,
        seed: int = 0,
):

    gradient_accumulation_steps = batch_size // micro_batch_size
    #load LLM mode & tokenizer
    model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
    #padding left eos
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"
    # load data
    train_data = PromptDataset(train_file=train_data_path,
                                    tokenizer=tokenizer, 
                                    max_len=cutoff_len,  
                                    sample=-1, 
                                    seed=seed, 
                                    category="Mix6Classes", 
                                    K = K)
   
    val_data = PromptDataset(train_file=eval_data_path, 
                                 tokenizer=tokenizer, 
                                 max_len=cutoff_len,  
                                 sample=2000, 
                                 category="Mix6Classes", 
                                 K = K)
    
    # generate huggingface dataset format for training
    # list[dict1,dict2,...] ->dict{input:[input1,...] output:[output1,...]}
    hf_train_dataset = HFDataset.from_dict(
        {k: [v[k] for v in train_data] for k in train_data[0].keys()}
        )
    hf_train_dataset = hf_train_dataset.shuffle(seed=seed)
    hf_val_dataset = HFDataset.from_dict(
        {k: [v[k] for v in val_data] for k in val_data[0].keys()}
        )
    #init trainer
    trainer = transformers.Trainer(
        model=model,
        train_dataset=hf_train_dataset,
        eval_dataset=hf_val_dataset,
        args=transformers.TrainingArguments(
            run_name=wandb_name,
            per_device_train_batch_size=micro_batch_size,
            per_device_eval_batch_size=micro_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_steps=200,
            num_train_epochs=num_epochs,
            learning_rate=learning_rate,
            bf16=True,
            logging_steps=1,
            optim="adamw_torch",
            max_steps=10000,
            eval_strategy="steps",       # Changed from "epoch" to "steps"
            eval_steps=2000,                   # Evaluate every 1000 steps
            save_strategy="steps",             # Changed from "epoch" to "steps"
            save_steps=2000,                   # Save checkpoint every 1000 steps

            output_dir=output_dir,
            save_total_limit=5,
            load_best_model_at_end=True,
            ddp_find_unused_parameters=None ,
            group_by_length=group_by_length,
            report_to="wandb",
        ),
        data_collator=transformers.DataCollatorForSeq2Seq(
            tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
        ),
        callbacks = [EarlyStoppingCallback(early_stopping_patience=5)],#
    )
    #training
    model.config.use_cache = False
    trainer.evaluate()
    trainer.train(resume_from_checkpoint=None)
    if not os.path.exists(output_dir):
            os.makedirs(output_dir)
    model.save_pretrained(output_dir)



if __name__ == "__main__":
    
    args = parse_args()
    
    train(
        base_model_path= args.base_model,
        train_data_path = args.train_data_path,
        eval_data_path= args.eval_data_path,
        output_dir=args.output_dir,
        wandb_name  = args.wandb_run_name
    )