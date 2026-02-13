#!/bin/bash

echo "start training"
MODEL_PATH=Qwen3-VL/Qwen3-VL-8B-Instruct  # replace it with your local file path
TRAIN_PARQUET=TRAIN.parquet  # replace it with your train parquet
TEST_PARQUET=TEST.parquet    # replace it with your test parquet


export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11 \
python3 -m verl.trainer.main \
    config=./examples/perception_config.yaml \
    data.train_files=${TRAIN_PARQUET} \
    data.val_files=${TEST_PARQUET} \
    data.mini_rollout_batch_size=96 \
    data.max_prompt_length=25000 \
    worker.actor.model.model_path=${MODEL_PATH} \
    worker.actor.clip_ratio_low=0.2 \
    worker.actor.clip_ratio_high=0.3 \
    worker.rollout.max_num_batched_tokens=28000 \
    algorithm.disable_kl=True \
    trainer.experiment_name=qwen3_vl_8b_perception \
    trainer.save_checkpoint_path=./verl_exp/qwen3_vl_8b_perception \
    trainer.n_gpus_per_node=12 \
    trainer.save_limit=10 \
    trainer.total_epochs=3
