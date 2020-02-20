## Variational Auto-Encoder

​	A simple implementation of [Variational Auto-Encoder](https://arxiv.org/abs/1312.6114) on MNIST dataset using Pytorch.

### Usage

​	`pip install -r requirements.txt`

​	`python -u main.py`

### Reference

	1. https://github.com/pytorch/examples/tree/master/vae

 	2. http://kvfrans.com/variational-autoencoders-explained/

### Update on 2/20/2020
&ensp; &ensp; Neither linear VAE nor convolutional VAE has good performance on CIFAR10... According to [this blog](https://medium.com/@joeDiHare/deep-bayesian-neural-networks-952763a9537), variational inference may be slow for deep Bayesian net and the performance is not guaranteed to be optimal.
