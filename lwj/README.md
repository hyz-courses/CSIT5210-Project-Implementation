# CSFT
## Git
```shell
git pull https://github.com/hyz-courses/CSIT5210-Project-Implementation.git 
git pull origin <remote_branch>:<local_branch>
```
push
```shell
git add . 
git commit -m "<message>"
git push origin <local_branch>:<remote_branch>
```
## train
GPU applition
```shell
srun --account=mscitsuperpod --partition=normal --gpus-per-node=1 --time=08:00:00 --pty bash
```
run csft train
```shell
bash run_CSFT.sh
```