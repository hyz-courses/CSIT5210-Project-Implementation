# First stage of LLM2Rec training -- Collaborative Supervised Fine-Tuning (CSFT).

model_path="/home/yingzhi/huggingface_data/hub/Qwen2-0.5B"  # Replace with your own model path

for category in "AmazonMix-6"
do
    train_file=$(ls -f ./data/${category}/5-core/train/${category}*.csv)
    eval_file=$(ls -f ./data/${category}/5-core/valid/${category}*.csv)
    echo ${train_file} ${info_file}

    # CUDA_VISIBLE_DEVICES=0,1 torchrun --master_port=25649 --nproc_per_node 2 \
    python    ./csft.py \
        --base_model ${model_path} \
        --train_data_path ${train_file} \
        --eval_data_path ${eval_file} \
        --output_dir ./output/Qwen2-0.5B-CSFT-${category} \
        --wandb_run_name Qwen2-0.5B-CSFT-${category} \
        # --category ${category} \
        

    # cp ${model_path}/*token* ./output/Qwen2-0.5B-CSFT-${category}/
    # Also copy tokenizer to the last checkpoint
    # latest_ckpt=$(ls -d ./output/Qwen2-0.5B-CSFT-${category}/checkpoint-* | sort -V | tail -n 1)
    # cp ${model_path}/*token* ${latest_ckpt}/
done

