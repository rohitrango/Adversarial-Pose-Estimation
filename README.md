# Adversarial Pose Estimation
## Abstract
This repository aims to replicate the results of [this](https://arxiv.org/pdf/1705.00389v2.pdf) paper. The idea is to augment the human pose estimation by using a GAN-based framework, where the (conditional) generator learns the distribution P(y|x), where x is the image and y is the heatmap for the person. Typical keypoint detectors simply employ a similarity based loss (MSE or cross-entropy) on the predicted heatmaps with the ground-truth heatmaps. However, these losses can predicted smooth outputs as they are averaged over the entire spatial domain. The idea here is to make the predictions ''crisper and sharper'' by employing discriminators that differentiate between ground-truth and predicted heatmaps in 2 different ways.

## Framework


## Dependencies
The list of dependencies can be found in the the `requirements.txt` file. Simply use `pip install -r requirements.txt` to install them.

## Running the code
Running the code for training is fairly easy. Follow these steps.

- Go to `config/default_config.py` to edit the hyperparameters as per your choice. For your convenience, the default parameters are already set. 
- Download the extended LSP dataset [here](http://sam.johnson.io/research/lspet.html). Download it in your favorite directory. Your dataset directory should look like this (if the root dataset dir is `lspet_dataset/`)
	```
	lspet_dataset/
		images/
			im00001.jpg
			im00002.jpg
			...
		joints.mat
		README.txt	
	```
- Add this path to the `--path` parameter in `train.sh` script. This contains all the other parameters required to train the model. 
- Run the script.
- The pretrained file can be found in the Downloads sections of the README.

## Results 
Coming soon.

## References
If you liked this repository, and would like to use it in your work, consider citing the original paper.
```
@article{DBLP:journals/corr/ChenSWLY17,
  author    = {Yu Chen and
               Chunhua Shen and
               Xiu{-}Shen Wei and
               Lingqiao Liu and
               Jian Yang},
  title     = {Adversarial PoseNet: {A} Structure-aware Convolutional Network for
               Human Pose Estimation},
  journal   = {CoRR},
  volume    = {abs/1705.00389},
  year      = {2017},
  url       = {http://arxiv.org/abs/1705.00389},
  archivePrefix = {arXiv},
  eprint    = {1705.00389},
  timestamp = {Mon, 13 Aug 2018 16:47:51 +0200},
  biburl    = {https://dblp.org/rec/bib/journals/corr/ChenSWLY17},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
``` 
We also thank [Naman's repository](https://github.com/Naman-ntc/Pytorch-Human-Pose-Estimation) for providing the code for PCK and PCKh metrics.

## Downloads
TODO: Put a drive link to pretrained model.