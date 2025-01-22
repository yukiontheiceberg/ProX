#!/bin/bash
#SBATCH --job-name=prox_doc_refining_xs
#SBATCH --partition=mbzuai
#SBATCH --nodes=1
#SBATCH --gres=gpu:8

source /mbz/users/yuqi.wang/miniconda3/bin/activate refining

export NGPU=8
export HF_TOKEN=
cmd="
for i in \$(seq 0 \$((NGPU-1))); do
    TOTAL_SPLIT=1 \\
    NODE_GPUS=1 \\
    NODE_RANK=${SLURM_NODEID} \\
    CUDA_VISIBLE_DEVICES=\$i \\
    python -m data_gen.tasks.apply_doc_refining_txt360 \\
        --data_format jsonl.gz \\
        --limit -1 \\
        --model_path gair-prox/web-doc-refining-lm \\
        --config_path data_gen/configs/apply_doc_refining.yaml \\
        --data_path $1 \\
        > /mbz/users/yuqi.wang/ProX/logging/apply_doc_refining_txt360_${SLURM_JOB_ID}_\${i}.log 2>&1 &
done
wait
"

echo "Executing command:"
echo "$cmd"

srun bash -c "$cmd"

# # ****************************************************
# # scripts for single node: (debug)
# # ****************************************************
# # setup env
# chmod +x setup_personal_env.sh
# chmod +x setup_common_env.sh
# source setup_personal_env.sh
# source setup_common_env.sh

# # activate conda env
# source $TINYLM_CONDA_DIR
# conda activate llama_factory

# # enter working dir
# cd $TINYLM_WORK_DIR

# export NNODE=1
# export NGPU=1
# # total split (int) = nnode * ngpu, write in shell expression
# export TOTAL_SPLIT=$((NNODE*NGPU))
# export SLURM_NODEID=0
# for i in $(seq 0 $((NGPU-1))); do
#   TOTAL_SPLIT=$TOTAL_SPLIT NODE_GPUS=$NGPU NODE_RANK=$SLURM_NODEID CUDA_VISIBLE_DEVICES=$i \
#   python -m data_gen.tasks.apply_doc_refining \
#     --data_format jsonl.gz \
#     --limit 1000 \
#     --model_path gair-prox/doc_refining_web_lm \
#     --config_path data_gen/configs/apply_doc_refining.yaml \
#     > ./logging/apply_doc_refining_${SLURM_NODEID}_${i}.log &
# done
