July 2016
Authors: Michael Mathieu, Camille Couprie

Update: due to large files that could not be stored on github, the trained models and dataset may be found at:
http://perso.esiee.fr/~coupriec/MathieuICLR16TestCode.zip

This repository contains:

- Test code for the ICLR 2016 paper:
[1] Michael Mathieu, Camille Couprie, Yann LeCun:
"Deep multi-scale video prediction beyond mean square error".
http://arxiv.org/abs/1511.05440
http://cs.nyu.edu/~mathieu/iclr2016.html

- Two trained models (using adversarial+l2norm training or
 adversarial+l1norm+gdl training).

- A subset of the UCF101 test dataset [2] with optical flow results to perform
an evaluation in moving area as described in [1].

- A training script for the model. Because the Sports1m dataset is hard to get,
we cannot provide an easy script to train on it. Instead, we propose a script
to train on UCF101, which is significantly smaller. 

Main files:
- For testing: test-frame-prediction-on-ucf-rec_gdl.lua
Script to test 2 trained models to predict future frames in video from 4
previous ones on a subset of the UCF101 test dataset.

- For training: - For training: train_iclr_model.lua
Script to train a model from scratch on the UCF101 dataset. If you want to
train on the Sports1m dataset, you will need to download it and write a
datareader, similar to datasources/ucf101.lua .

Usage:

1- Install torch and the packages (standard packages + nngraph, cudnn.torch, gfx.js)

For testing:
2- Uncompress the provided archives.
3- Run the main script :
th test-frame-prediction-on-ucf-rec_gdl.lua

It generates results (2 predicted images + animated gifs)
in a directory named 'AdvGDL'.
It also display the average PSNR and SSIM of the 2 first predicted frames
following the evaluation presented in [1].

For training:
2- Get the UCF101 dataset (requires unrar, modify the script if you have another .rar extractor):
cd datasources
python get_datasource.py
3- Get thffpmeg from https://github.com/MichaelMathieu/THFFmpeg
4- Run the training script:
th train_iclr_model.lua
5- For visualizing the intermediate results, start the gfx.js server
th -lgfx.start
And go to http://localhost:8000 in your internet browser.

[2]:Khurram Soomro, Amir Roshan Zamir and Mubarak Shah,
UCF101: A Dataset of 101 Human Action Classes From Videos in The Wild.,
CRCV-TR-12-01, November, 2012.




