srun --account=mscitsuperpod --partition=normal --gpus-per-node=1 --time=08:00:00 --pty bash

CUDA_VISIBLE_DEVICES=0 /home/$USER/llm2rec-venv/bin/torchrun \
    --master_port=25649 --nproc_per_node 1 \
    -m train_LLM.train_csft 

exit