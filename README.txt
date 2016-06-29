July 2016
Authors: Michael Mathieu, Camille Couprie

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

Main file: test-frame-prediction-on-ucf-rec_gdl.lua

Script to test 2 trained models to predict future frames in video from 4
previous ones on a subset of the UCF101 test dataset.

Usage:

1- Install torch and the packages.
2- Uncompress the provided archives.
3- Run the main script :
th test-frame-prediction-on-ucf-rec_gdl.lua

It generates results (2 predicted images + animated gifs)
in a directory named 'AdvGDL'.
It also display the average PSNR and SSIM of the 2 first predicted frames
following the evaluation presented in [1].


[2]:Khurram Soomro, Amir Roshan Zamir and Mubarak Shah,
UCF101: A Dataset of 101 Human Action Classes From Videos in The Wild.,
CRCV-TR-12-01, November, 2012.




