#!/bin/bash

srun --pty --nodes=1 --ntasks=1 --cpus-per-task=16 -G 1 --time=04:00:00 --partition=short  --container-image /work/group/humingamelab/sqsh_images/pytorch.sqsh --container-mounts="${HOME}"/Projects/forks/pytorch-segmentation:/workspace,/work/users/ejlaird/data/image_data:/data --container-workdir=/workspace /bin/bash