##Various Instructions:
**[Pytorch installation on Stanford Clusters](pytorch-installation-on-stanford-clusters)**
**[Generating Results](generating-results)**

### PyTorch installation with GPU access on Stanford clusters

login to rice:
`ssh [sunetid]@rice.stanford.edu`

download anaconda install script, make exectutable, and run:

`wget https://repo.continuum.io/archive/Anaconda3-5.0.1-Linux-x86_64.sh`

`chmod +x Anaconda3-5.0.1-Linux-x86_64.sh`

`./Anaconda3-5.0.1-Linux-x86_64.sh`
 
install pytorch for python 3.6, CUDA 8.0:

`conda install pytorch torchvision cuda80 -c soumith`

enter `yes` when asked for permission to edit bash script

Now we download a test script in this repo and run it. The files must be downloaded to a location outside afs-home, as it seems the GPU cannot access files inside, due to incompatible file systems. So just download it to /home/[username]/:

`cd /home/[username]/`

`git clone https://github.com/marcuspanje/cell_segmentation`

To test pytorch with GPU, we will run the file testTorch.py, edited from  [here](http://pytorch.org/tutorials/beginner/pytorch_with_examples.html#pytorch-tensors).
To get GPU acess, we need to submit the files to the oat cluster. More info is [here](https://web.stanford.edu/group/farmshare/cgi-bin/wiki/index.php/FarmShare_2#oat.stanford.edu).
We first make the python file executable:

`chmod +x testTorch.py`

then submit the command to be run on oat:

`srun --partition=gpu --gres=gpu:1 ./testTorch.py`

If everything works, you should see a printout of iteration step and losses, as documented in the `testTorch.py`:

0 40479688.0

1 40655632.0

...

### Generating Results
Edit `generateResults.py` generates a latex document with the image results. 
The first 2 columns are the raw image and ground truth. 
The next 2 columns are 2 image result sources. Set the `prefix1` and `prefix2` 
to the folder containing the images. Then run:
`python generateResults.py`
`pdflatex results.tex`
The `results.pdf` should be produced.
