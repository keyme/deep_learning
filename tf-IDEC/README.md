# DEC/IDEC
---
The modules in this directory implement:

* [DEC](http://proceedings.mlr.press/v48/xieb16.pdf) (Unsupervised Deep Embedding for Clustering Analysis) and
* [IDEC](https://xifengguo.github.io/papers/IJCAI17-IDEC.pdf) (Xifeng Guo, Long Gao, Xinwang Liu, Jianping Yin. Improved Deep Embedded Clustering with Local Structure Preservation. IJCAI 2017)

Everything is written in pure `tensorflow`/`python` on only the MNIST dataset. Special thanks to the authors for providing a Keras implementation [here](https://github.com/XifengGuo/IDEC), which we base a lot of this on. Testing error in our implementation converges between 87-88%.

The key differences between our implementation and the paper are as follows:
* We train our autoencoder end-to-end; that is, we do not do greedy, layer-wise training.
* We used Adam as the optimizer for both pretraining and training.

