
# Template-based-Transformation

This repository contains the official PyTorch implementation of Templated-based-Transformation work.


*****

### Requirements
* Python 3.8
* Pytorch 2.1.1
* torchvision 0.15.2 

To install requirements with pip:
<pre>
<code>
pip install -r requirements.txt
</code>
</pre>
*****

### Training
* exp-name=c_mnist (Color MNIST)
* exp-name=mnist (MNIST)
* exp-name=fmnist (Fashion MNIST)
* exp-name=svhn (SVHN)
* exp-name=cifar (cifar10)
* exp-name=fc100 (cifar100) 
* exp-name=cub200 (cub200) # you should download data from the website


To train the model in the paper, run the following command :
<pre>
<code>
python proposed_train.py --data-num=10 --exp-name=svhn 
</code>
</pre>
It will save the trained model in the epoch_result directory.

### Image Synthesis by Shuffling
To synthesize new data by shuffling, run the following command (parameter must be the same as training):

<pre>
<code>
python draw_shuffle_fig.py --data-num=10 --exp-name=svhn --target-class=3
</code>
</pre>
It will save the figure in the fig directory.

### Data Augmentation
To perform data augmentation, run the following commands :

<pre>
<code>
python proposed_inference.py --data-num=10 --exp-name=svhn
python data_aug_experim.py --data-num=10 --exp-name=svhn --aug-policy=proposed
</code>
</pre>
The first line will save generated images in epoch_result folder.
The second line will perform data augmentation experiment