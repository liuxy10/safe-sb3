# examples
# link checkpoints 
ln -s /home/vision/src/data/metadrive/saved_bc_policy /home/vision/src/xinyi/safe-sb3/examples/metadrive/saved_policy
# copy h5py to github for visualization on pc
cp  /home/vision/src/data/metadrive/h5py/bc_9_900.h5py /home/vision/src/xinyi/safe-sb3/examples/metadrive/h5py/bc_9_900.h5py
# copy checkpoint model to github for visualization on 
cp  /home/vision/src/data/metadrive/saved_bc_policy/bc-waymo-es0_120000_steps.zip /home/vision/src/xinyi/safe-sb3/examples/metadrive/saved_bc_policy/bc-waymo-es0.zip


# shh copy checkpoint from server to pc
scp vision@128.32.164.115:/home/vision/src/data/metadrive/saved_sac_policy/heading_acc/sac-waymo-es0_770000_steps.zip /home/xinyi/Documents/UCB/safe-sb3/examples/metadrive/saved_bc_policy/

# ssh copy model.pt from server to pc

# ssh copy downloaded mojuco to server
scp /home/xinyi/Documents/UCB/safe-sb3/examples/metadrive/saved_bc_policy/ vision@128.32.164.115:/home/vision/src/data/metadrive/saved_sac_policy/heading_acc/sac-waymo-es0_770000_steps.zip 