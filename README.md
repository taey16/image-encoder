# image-encoder
This is a train / evaluating(inference) system for vision-networks which is originally posted on [imagenet-multiGPU.torch](https://github.com/soumith/imagenet-multiGPU.torch)

# Features
- includes inference code fragments with threading
- includes code fragments for load vgg16 from Caffe model zoo (with [loadCaffe](https://github.com/szagoruyko/loadcaffe))
- includes various models such as inception5~7 [Rethinking the Inception Architecture for Computer Vision](http://arxiv.org/abs/1512.00567)
- includes residual learning idea in our inception7 [Deep Residual Learning for Image Recognition](http://arxiv.org/abs/1512.03385)
- includes training with Batch-normalizaation(BN; around 5.5 days reaching 69% on ILSVRC2012 val. set(single-crop)) [Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift](http://arxiv.org/abs/1502.03167)
- includes absorbing BN parameters into convolutional parameter(in inference(prediction) step, all of the nn.(Spatial)BatchNormalization layers is to be removed so that elapsed time is impressively reduced)
