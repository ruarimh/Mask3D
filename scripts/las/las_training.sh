#!/bin/bash
export OMP_NUM_THREADS=3

CURR_DBSCAN=14.0
CURR_TOPK=10
CURR_QUERY=160
CURR_SIZE=250

EXPERIMENT_NAME="all_plots_8.0_voxel_newotherclass_8batch_500crop_8batch_500epoch"
SAVE_DIRECTORY="/content/drive/MyDrive/MRes/saved/"

# TRAIN
python main_instance_segmentation.py \
general.experiment_name=${EXPERIMENT_NAME} \
general.project_name="las" \
data/datasets=las \
general.num_targets=2 \
data.num_labels=2 \
data.voxel_size=8.0 \
data.num_workers=0 \
data.cache_data=true \
data.cropping_v1=false \
data.cropping=false \
data.batch_size=8 \
data.test_batch_size=1 \
general.reps_per_epoch=5 \
model.num_queries=${CURR_QUERY} \
general.use_dbscan=false \
general.on_crops=true \
model.config.backbone._target_=models.Res16UNet18B \
data.crop_length=${CURR_SIZE} \
data.subplot_size=500.0 \
general.eval_inner_core=-1 \
general.save_visualizations=false \
general.save_dir="${SAVE_DIRECTORY}${EXPERIMENT_NAME}" \
trainer.check_val_every_n_epoch=10 \
trainer.max_epochs=500