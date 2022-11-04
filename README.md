# CIFAR10-Pytorch
> For my homework \
Creating your own github account. \
Implementing your own deep neural network (in Pytorch, PaddlePaddle…). \
Training it on CIFAR10.\
Tuning a hyper-parameter and analyzing its effects on performance.
\
Writing a README.md to report your findings.
	Example： https://github.com/StanfordVL/taskonomy/tree/master/taskbank

## Train Model
> python main.py --shortcut_level shortcut-level

The training and validation log will be saved in the './logs' folder.

## Model Innovation
This model adds more residual structures based on the ResNet18. I think the residual structure not only can solve the problem of gradient explode, but also can combine the multiscale features, which will help the model to complete the task of classification better. In fact, this modification imporoved the performance of ResNet model on the CIFAR-10 dataset indeedly.