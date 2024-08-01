#!/usr/bin/env zsh
#SBATCH -J res_seg
#SBATCH -o output/res_seg_%j.out
#SBATCH -p batch
#SBATCH -c 16 --mem=32G     
#SBATCH --nodes=1
#SBATCH -G 1
#SBATCH --time=1-20:00:00
#SBATCH --ntasks-per-node=1
#SBATCH --mail-user ejlaird@smu.edu
#SBATCH --mail-type=ALL     

MODE="resolution"
RESUME="./saved/DeepLabV3Plus/07-31_19-13/checkpoint-epoch20.pth"

save_dir="/scratch/users/ejlaird/dctresolution/${SLURM_JOB_ID}_res_seg"
mkdir -p ${save_dir}/image_data

echo "Saving data to ${save_dir}/image_data"


cleanup() {
    echo "Removing ${save_dir} from SCRATCH"
    rm -rf ${save_dir}
    echo "Removed ${save_dir} from SCRATCH"
}

trap cleanup SIGTERM SIGINT EXIT


echo "Copying data to ${save_dir}"
cp -r /work/users/ejlaird/data/image_data/VOCdevkit ${save_dir}/image_data
echo "Data copied to ${save_dir}"

echo $(ls ${save_dir})

echo "Running segmentation"
srun --no-container-entrypoint --partition=short --container-image /work/group/humingamelab/sqsh_images/pytorch.sqsh --container-mounts="${HOME}"/Projects/forks/pytorch-segmentation:/workspace,${save_dir}/image_data:/data --container-workdir=/workspace bash -c "./entrypoint.sh ${MODE} ${RESUME}"
