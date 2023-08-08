# conda activate metadrive-gym
cd ~/src/data/metadrive/tfrecord_9

# Define the range of numbers you want to iterate over
start=1
end=50
# Iterate over the range of numbers
for ((i=start; i<=end; i++)); do
    # Format the number with leading zeros to ensure it has 5 digits
    formatted_number=$(printf "%05d" "$i")

    # Create a file name using the formatted number
    wb_path="gs://waymo_open_dataset_motion_v_1_1_0/uncompressed/scenario/training/training.tfrecord-${formatted_number}-of-01000"


    gcloud storage cp $wb_path .

done

# gcloud storage cp gs://waymo_open_dataset_motion_v_1_1_0/uncompressed/scenario/training/training.tfrecord-00001-of-01001 .