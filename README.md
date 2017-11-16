# cell_segmentation

## DRN training
python3 segment.py train -d [dataset directory] -c 3 -s 360 --arch drn_d_22 --batch-size 16 --epochs 250 --lr 0.001 --momentum 0.99 --step 100

-c: number of classes \
-s: crop size for data augmentation (default value of 0 was leading to an error) \
--arch: currently using the D 22 architecture (See drn git for other options https://github.com/fyu/drn)

Uses train_images.txt, train_labels.txt, val_images.txt, val_labels.txt (which currently point to files in /NKI_Training/labeled_preprocess/ or /VGH_Training/labeled_preprocess/)

preprocessingSave.py: contains script to threshold and save jpg images to the labeled_preprocess folders
