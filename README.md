# Seismic-Velocity-Inversion-Based-on-Deep-Convolutional-Neural-Networks

## background
With the continuous exploration and mining of underground oil and natural gas resources, the exploration of the easy production area is constantly limited, which makes it more difficult to carry out more complex region of exploration work, and it becomes a more serious challenge for earthquake exploration technology. The core of seismic exploration is the seismic wave, so accurate speed model is an important part of seismic exploration technology. However, the traditional methods are constrained, time consuming and costly, and they seriously rely on people's interaction. Introducing deep learning into the field of seismic exploration, which can significantly reduce a large number of repetitive work in traditional methods with quick speed and high efficiency.

According to this thesis, the problem of determining the underground velocity parameters can be converted to a multi-output regression problem from seismic signals to velocity. The experiment generates seismic data by spontaneously generated seismic signals based on the seismic section data and treats them as input and output. Based on the CNN, the thesis proposes two algorithms called UnetRes1 and UnetRes2, which implements U-net with Residual Network and applies an end-to-end supervised learning to the synthetic seismic data and ground-truth velocity model, and the velocity model is established directly from the original seismic signal graph. In the training phase, the network establishes a nonlinear mapping from multi-seismic data to the corresponding velocity model. In the test phase, the training completed networks can be used to estimate the velocity model of new input seismic data.

This thesis concludes that the performance of UnetRes2 is better than UnetRes1 and U-net, and performance of UnetRes1 is much worse than UnetRes2. This is because that U-net can extract the features of pictures through the compression and expansion paths, skip connections, etc. and retain the information of original images. Adding residual learning to each convolutional layers can, to some extent, reduce training loss to avoid training degradation with the increase of the numbers of convolutional layers, and the more numbers of convolutional layers that the input skips in the residual network, the better performance of the network is. Therefore, once a good generalized network is established, the calculation time of the seismic inversion can be greatly reduced.

## data
To train an effective neural network, an appropriate large-scale training set consisting of input-output pairs is necessary. In this paper, the input data set is obtained using seismic exploration methods, which involve exciting seismic waves at a particular location along a straight line on the ground and using detectors to receive the seismic signals that have propagated underground due to the reflection or refraction of sound waves. These signals can be used to determine underground parameters such as velocity and other parameters that need to be solved for seismic data processing. Based on the observed data and determined underground parameters, the geological properties and structures within the Earth can be inferred. By emitting seismic waves from controllable source locations, the different propagation speeds through different geological layers can be simulated through velocity modeling to simulate the underground structure.

The training data set consists of 1600 input-output pairs, with the output being the true seismic profile that shows the different propagation speeds of seismic waves in different underground structures. The input consists of seismic data, and for each velocity model, five sources are uniformly placed to simulate a shot gather. Additionally, 301 receivers are uniformly placed at the same spatial intervals to form the geometric structure. A velocity model is generated as input data by using the wave equation based on the original seismic profile.

Since an end-to-end supervised learning method is used, the ground truth velocity models in the test data set have similar geological structures to the training data set. All velocity models used for prediction are not included in the training data set and are unknown during the prediction process. The input seismic data used for prediction is also obtained using the same method as the input data used to generate the training data set. The test data set consists of 100 samples.

## method - network structure
![Unet+ResNet](https://github.com/Yiran-H/Seismic-Velocity-Inversion-Based-on-Deep-Convolutional-Neural-Networks/blob/main/UnetRes2.png)
<br>[model code](https://github.com/Yiran-H/Seismic-Velocity-Inversion-Based-on-Deep-Convolutional-Neural-Networks/blob/main/func/UnetRes2.py)

## report
[chinese version](https://github.com/Yiran-H/Seismic-Velocity-Inversion-Based-on-Deep-Convolutional-Neural-Networks/blob/main/undergraduate_thesis.pdf)

## paper
[chinese version](https://github.com/Yiran-H/Seismic-Velocity-Inversion-Based-on-Deep-Convolutional-Neural-Networks/blob/main/20220712_O_origin_reject.pdf)
(it was reject though)

## citation
all the work is based on the [FCNVMB](https://github.com/YangFangShu/FCNVMB-Deep-learning-based-seismic-velocity-model-building)
