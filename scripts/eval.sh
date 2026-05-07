if [ -x ".venv/bin/python" ]; then
    PYTHON=".venv/bin/python"
elif command -v python >/dev/null 2>&1; then
    PYTHON="python"
else
    PYTHON="python3"
fi

EXPERIMENT_NAME="radii_one_step_init"
DATA_ROOT="data/radii/radii.hdf5"
DATA_CATEGORIES="['radii']"
CHECKPOINT_PATH="output/finetune_radii_2026-04-17_10-40-00/GARF/k8bs0qdp/checkpoints/epoch=19-step=840.ckpt"
BASE_CHECKPOINT_PATH="models/GARF_mini.ckpt"
# With batch_size=1, this is the number of validation samples to evaluate.
NUM_VAL_SAMPLES=3

HYDRA_FULL_ERROR=1 "$PYTHON" eval.py \
    seed=42 \
    experiment=denoiser_flow_matching \
    "experiment_name='$EXPERIMENT_NAME'" \
    loggers=csv \
    loggers.csv.save_dir=logs/GARF \
    trainer.num_nodes=1 \
    trainer.devices=[0] \
    +trainer.limit_test_batches=$NUM_VAL_SAMPLES \
    data.data_root=$DATA_ROOT \
    data.categories=$DATA_CATEGORIES \
    data.batch_size=1 \
    base_ckpt_path=$BASE_CHECKPOINT_PATH \
    "ckpt_path='$CHECKPOINT_PATH'" \
    ++data.random_anchor=false \
    ++model.inference_config.one_step_init=true
