# Meta Networks (MetaNet) #

This project contains Chainer implementation of [Meta Networks](https://arxiv.org/abs/1703.00837) and imagenet samples (mini-imagenet images) used in the experiments. 

- Paper: [https://arxiv.org/abs/1703.00837](https://arxiv.org/abs/1703.00837)

![metanet](./assets/metanet.png)

MetaNet is a flexible and modular NN architecture that uses fast weights and a layer augmentation method to modify synaptic connection (weights) in NNs and performs self-adaptation to new task or environment on the fly. MetaNet can be seen a framework for building flexible models:

- Step 1: Define a NN architecture for your problem
- Step 2: Define the adaptable part of the NN
- Step 3: Define your feedback in form of meta info
> The loss gradient works, but expensive!
- Step 4: Fast-parameterize and layer-augment
> Inject the fast weights and update the synaptic connections!
- Step 5: Optimize your model end-to-end






Prerequisites
-------------

- Python 2.7
- [chainer](http://chainer.org/) (tested on chainer 1.19.0)
- Other data utils: sklearn, numpy etc.


Data
-----
Please check out data/miniImagenet/ for the exact imagenet samples used in the experiments. For preprocessing of miniImagenet data, please see preprocess_mini_imagenet.py script (all you need to do is to resize the images).

Omniglot data is available at https://github.com/brendenlake/omniglot.

Usage
-----
Once the data is preprocessed and stored in npz format, set input_path to the path of the data dir. Run "python train_mini_imagenet.py or train_omniglot.py" to start training and test.

The model is very flexible. Feel free to change the parameters and play around. The parameters include the number of outputs during training: n_outputs_train, # of outputs during test: n_outputs_test, # of support examples: nb_samples_per_class and so on. With n_outputs_train and n_outputs_test, you can switch between different softmax outputs, feel free to try out more than two softmax outputs. You can also set different values for # of support examples (nb_samples_per_class) during training and test, i.e. you can train the model on one-shot task, then test it on n-shot (3-shot, 5-shot etc.) tasks. Supplying more support examples during test increases the test performance as well.


Results
-------
MetaNet achieved SOTAs on Omniglot and miniImagenet one-shot tasks. It also demonstrated some interesting properties relating to generalization and continual learning. Please check out our paper for detailed insights.



Author
------
Tsendsuren Munkhdalai / [@tsendeemts](http://www.tsendeemts.com/)