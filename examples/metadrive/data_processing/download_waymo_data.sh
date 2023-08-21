# install google cloud cli
# curl -O https://dl.google.com/dl/cloudsdk/channels/rapid/downloads/google-cloud-cli-442.0.0-linux-x86_64.tar.gz

# conda activate metadrive-gym
cd ~/src/data/metadrive/tfrecord_9

# Define the range of numbers you want to iterate over
start=0
end=2
# Iterate over the range of numbers
for ((i=start; i<end; i++)); do
    # Format the number with leading zeros to ensure it has 5 digits
    formatted_number=$(printf "%05d" "$i")

    # Create a file name using the formatted number
    wb_path="gs://waymo_open_dataset_motion_v_1_1_0/uncompressed/scenario/training/training.tfrecord-${formatted_number}-of-01000"


    gcloud storage cp $wb_path .

done

