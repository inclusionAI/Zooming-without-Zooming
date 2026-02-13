echo "start training"
NNODES=8
MODEL_PATH=Qwen3-VL/Qwen3-VL-8B-Instruct  # replace it with your local file path
TRAIN_PARQUET=TRAIN.parquet
TEST_PARQUET=TEST.parquet
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python3 -m verl.trainer.main \
    config=./examples/perception_config.yaml \
    data.train_files=${TRAIN_PARQUET} \
    data.val_files=${TEST_PARQUET} \
    data.mini_rollout_batch_size=128 \
    data.max_prompt_length=25000 \
    data.max_response_length=2048 \
    data.rollout_batch_size=512 \
    data.val_batch_size=256 \
    worker.actor.global_batch_size=128 \
    worker.actor.model.model_path=${MODEL_PATH} \
    worker.actor.clip_ratio_low=0.2 \
    worker.actor.clip_ratio_high=0.3 \
    worker.rollout.max_num_batched_tokens=28000 \
    worker.reward.reward_function=./examples/reward_function/perception_multinode.py:compute_score \
    algorithm.disable_kl=True \
    trainer.experiment_name=qwen3_vl_8b_perception_multinode \
    trainer.save_checkpoint_path=./verl_exp/qwen3_vl_8b_perception_multinode \
    trainer.save_limit=100 \
    trainer.n_gpus_per_node=16 \
    trainer.nnodes="${NNODES}" \
    trainer.total_epochs=4