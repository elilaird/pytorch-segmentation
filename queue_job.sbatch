#!/usr/bin/env zsh
#SBATCH -J res_seg
#SBATCH -o output/res_seg_%j.out
#SBATCH -p short
#SBATCH -c 16 --mem=32G     
#SBATCH --nodes=1
#SBATCH -G 1
#SBATCH --time=0-04:00:00
#SBATCH --ntasks-per-node=1
#SBATCH --mail-user ejlaird@smu.edu
#SBATCH --mail-type=ALL     

srun --no-container-entrypoint --partition=short --container-image /work/group/humingamelab/sqsh_images/pytorch.sqsh --container-mounts="${HOME}"/Projects/forks/pytorch-segmentation:/workspace,/work/users/ejlaird/data/image_data:/data --container-workdir=/workspace bash -c "./entrypoint.sh"
