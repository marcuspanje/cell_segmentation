All code can also be accessed throught the github link - https://github.com/marcuspanje/cell_segmentation

Deep-Learning Model files:
fcn32stats.py
u_net.py

Training file:
train_unet.py
trainStats.py

Testing file:
evaluate.py

Generating confusion matrices:
confusion_matrix.py

Saved models (after training):
in the saved_models folder

For FCN/Unet we ran on GPUs on rice.stanford.edu using the command "sbatch submit.sh". We changed the submit.sh script depending on whether we were training or testing.

Instructions for training DRN (done on Google Cloud):

preprocessingSave.py: contains script to threshold and save jpg images to the labeled_preprocess folders 
Change data directory and run preprocessing script to save modified label jpg files in the labeled_preprocess folders (have to change 'phase' each time to and run on train, val, test)

python3 segment.py train -d [dataset directory] -c 3 -s 360 --arch drn_d_22 --batch-size 16 --epochs 250 --lr 0.001 --momentum 0.99 --step 100

-c: number of classes 
-s: crop size for data augmentation (default value of 0 was leading to an error) 
--arch: currently using the D 22 architecture (See drn git for other options https://github.com/fyu/drn)

Uses train_images.txt, train_labels.txt, val_images.txt, val_labels.txt (which currently point to files in /NKI_Training/labeled_preprocess/ or /VGH_Training/labeled_preprocess/)
