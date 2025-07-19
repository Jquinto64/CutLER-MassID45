#!/bin/bash
#SBATCH -p rtx6000           # partition: should be gpu on MaRS, and a40, t4v1, t4v2, or rtx6000 on Vaughan (v)
#SBATCH --gres=gpu:1    # request GPU(s)
#SBATCH -c 8              # number of CPU cores
#SBATCH --mem=16G           # memory per node
#SBATCH --array=0           # array value (for running multiple seeds, etc)
#SBATCH --qos=m2
#SBATCH --time=8:00:00
#SBATCH --output=slogs/%x_%A-%a_%n-%t.out
                            # %x=job-name, %A=job ID, %a=array value, %n=node rank, %t=task rank, %N=hostname
                            # Note: You must manually create output directory "slogs" 
#SBATCH --open-mode=append  # Use append mode otherwise preemption resets the checkpoint file
#SBATCH --job-name=cutler_cascade_zero_shot_sahi_inference_2x_zoom_SR_real_esrgan_v9_025_conf_final_val_set_results_0.6_overlap
#SBATCH --exclude=gpu177

source ~/.bashrc
source activate cutler
module load cuda-11.3

SEED="$SLURM_ARRAY_TASK_ID"

# Debugging outputs
pwd
which conda
python --version
pip freeze

########################################### RUNNING ####################################################################################33
TILE_SIZE=512
python sahi_inference.py --model_path /scratch/ssd004/scratch/jquinto/CutLER/cutler_cascade_final.pth \
--exp_name cutler_zero_shot_cascade_${TILE_SIZE}_tiled_TEST_SET \
--dataset_json_path /h/jquinto/lifeplan_b_v9_cropped_center/annotations/instances_test2017.json \
--dataset_img_path /h/jquinto/lifeplan_b_v9_cropped_center/test2017 \
--config_path /scratch/ssd004/scratch/jquinto/CutLER/cutler/model_zoo/configs/CutLER-ImageNet/cascade_mask_rcnn_R_50_FPN.yaml \
--crop_fac 16 \
--postprocess_match_threshold 0.5 \
--model_confidence_threshold 0.25 \
--predict \
--scale_factor 1 \
--slice_height ${TILE_SIZE} \
--slice_width ${TILE_SIZE} \
--overlap 0.6
