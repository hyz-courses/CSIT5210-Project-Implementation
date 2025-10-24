model_path="/home/$USER/huggingface_data/hub/Qwen2-0.5B"  # Replace with your own model path

# Stage 2 - Train MNTP
# echo "Starting Stage 2 - Train MNTP..."
# CUDA_VISIBLE_DEVICES=0 /home/$USER/llm2rec-venv/bin/torchrun --nproc_per_node=1 --master_port=29501 ./llm2rec/run_mntp.py ./llm2rec/train_mntp_config.json
echo "Starting Stage 2 - Train MNTP..."
CUDA_VISIBLE_DEVICES=0 /home/$USER/llm2rec-venv/bin/torchrun --nproc_per_node=1 --master_port=29501 ./revised/mntp.py ./llm2rec/train_mntp_config.json
