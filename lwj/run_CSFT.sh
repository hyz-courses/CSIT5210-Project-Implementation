# First stage of LLM2Rec training -- Collaborative Supervised Fine-Tuning (CSFT).

base_model_path="/home/$USER/huggingface_data/hub/Qwen2-0.5B"  # Replace with your own model path
# data path
train_data_path=$(ls -f /home/$USER/code/CSIT5210-Project-Implementation/data/grained/AmazonMix/train_AmazonMix.csv)
eval_data_path=$(ls -f /home/$USER/code/CSIT5210-Project-Implementation/data/grained/AmazonMix/valid_AmazonMix.csv)
echo ${train_data_path} ${eval_data_path}

CUDA_VISIBLE_DEVICES=0 /home/$USER/llm2rec-venv/bin/torchrun --master_port=25649 --nproc_per_node 1 \
    ./csft.py \
    --base_model ${base_model_path} \
    --train_data_path ${train_data_path} \
    --eval_data_path ${eval_data_path} \
    --output_dir ./output/Qwen2-0.5B-CSFT-AmazonMix-CSIT5210G1 \
    --wandb_run_name Qwen2-0.5B-CSFT-AmazonMix-CSIT5210G1 \
    # --category ${category} \
    

cp ${model_path}/*token* ./output/Qwen2-0.5B-CSFT-AmazonMix-CSIT5210G1/
# Also copy tokenizer to the last checkpoint
# latest_ckpt=$(ls -d ./output/Qwen2-0.5B-CSFT-AmazonMix-6/checkpoint-* | sort -V | tail -n 1)
# cp ${model_path}/*token* ${latest_ckpt}/
