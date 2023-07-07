conda activate metadrive-gym
cd ~/src/data/metadrive/tfrecord_9
gcloud storage cp gs://waymo_open_dataset_motion_v_1_1_0/uncompressed/scenario/training/training.tfrecord-00000-of-01001 .
gcloud storage cp gs://waymo_open_dataset_motion_v_1_1_0/uncompressed/scenario/training/training.tfrecord-00001-of-01001 .