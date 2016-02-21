# image-encoder
This is a train / inference(prediction) system for vision-networks which is originally posted on [imagenet-multiGPU.torch](https://github.com/soumith/imagenet-multiGPU.torch)

# Features
- includes inference code fragments with threading
- includes code fragments for load vgg16 from Caffe model zoo (with [loadCaffe](https://github.com/szagoruyko/loadcaffe))
- includes code fragments for Kaiming initialization [Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification](http://arxiv.org/abs/1502.01852)
- includes various models such as inception-v1~v3 [Rethinking the Inception Architecture for Computer Vision](http://arxiv.org/abs/1512.00567)
- includes residual learning idea in the inception-v3 [Deep Residual Learning for Image Recognition](http://arxiv.org/abs/1512.03385)
- includes training with Batch-normalizaation(BN; around 5.5 days reaching 69% on ILSVRC2012 val. set(BN-inception5, single-crop)) [Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift](http://arxiv.org/abs/1502.03167)
- includes absorbing BN parameters into convolutional parameter(in inference(prediction) step, all of the nn.(Spatial)BatchNormalization layers is to be removed so that elapsed time is impressively reduced)
  [How does it works?](https://github.com/taey16/image-encoder/blob/master/example/logs/BN-absorb_derivation.png)
- In our experiment, best accuracy was reached around top1: ~75% on ILSVRC2012 val. set with single-crops, single-scale, [resception-net](https://github.com/taey16/image-encoder/blob/master/models/resception.lua)
- The google brain team's experiment result on inception-ResNet, inception-v4: [open-review, ICLR, 2016](http://beta.openreview.net/pdf?id=q7kqBkL33f8LEkD3t7X9)
- cudnn-v4 supports (in our case, 1.6x faster in conv. than cudnn-v3's conv. with cudnn.fastest=true, cudnn.benchmark=true)

# Acknowledgements
- Soumith's great works [imagenet-multiGPU.torch](https://github.com/soumith/imagenet-multiGPU.torch)
- Elad Hoffer's great works [ImageNet-Training](https://github.com/eladhoffer/ImageNet-Training)
- e-lab@Purde Univ. [torch-toolbox](https://github.com/e-lab/torch-toolbox)  
- [cudnn-v4, cudnn.torch](https://github.com/soumith/cudnn.torch)
- [FAIR's ResNet with multi-gpu, cudnn-v4](http://torch.ch/blog/2016/02/04/resnets.html)

Feel free to e-mail taey1600@gmail.com
