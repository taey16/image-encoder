
require 'cudnn'
require 'cunn'
paths.dofile('inception_module.lua')


-- Batch Normalization - Accelerating Deep Network Training by Reducing Internal Covariate Shift, arXiv, 2015
-- refer to Appendix (Figure 5: Inception architecture)
function createModel()
  local feature = nn.Sequential() 
  -- 224
  feature:add(cudnn.SpatialConvolution(3, 64, 7, 7, 2, 2, 3, 3))
  feature:add(nn.SpatialBatchNormalization(64))
  feature:add(cudnn.ReLU(true))
  feature:add(cudnn.SpatialMaxPooling(3, 3, 2, 2, 1, 1))
  -- 56
  feature:add(cudnn.SpatialConvolution(64,64, 3, 3, 1, 1, 1, 1))
  feature:add(nn.SpatialBatchNormalization(64))
  feature:add(cudnn.ReLU(true))
  feature:add(cudnn.SpatialConvolution(64,192, 3, 3, 1, 1, 1, 1))
  feature:add(nn.SpatialBatchNormalization(192))
  feature:add(cudnn.ReLU(true))
  feature:add(cudnn.SpatialMaxPooling(3, 3, 2, 2, 1, 1))
  -- 28
  feature:add(inception_module(2, 192, 1, {{ 64}, {64,  64}, {64,  96}, {'avg', 32}})) --3a
  feature:add(inception_module(2, 256, 1, {{ 64}, {64,  96}, {64,  96}, {'avg', 64}})) --3b
  feature:add(inception_module(2, 320, 2, {{  0}, {128,160}, {64,  96}, {'max',  0}})) --3c, stride2
  -- 14
  feature:add(inception_module(2, 576, 1, {{224}, {64,  96}, {96, 128}, {'avg',128}})) --4a
  feature:add(inception_module(2, 576, 1, {{192}, {96 ,128}, {96, 128}, {'avg',128}})) --4b
  feature:add(inception_module(2, 576, 1, {{160}, {128,160}, {128,128}, {'avg',128}})) --4c
  feature:add(inception_module(2, 576, 1, {{ 96}, {128,192}, {160,160}, {'avg',128}})) --4d
  feature:add(inception_module(2, 576, 2, {{  0}, {128,192}, {192,256}, {'max',  0}})) --4e, stride2
  -- 7
  feature:add(inception_module(2,1024, 1, {{352}, {192,320}, {160,224}, {'avg',128}})) --5a
  feature:add(inception_module(2,1024, 1, {{352}, {192,320}, {192,224}, {'avg',128}})) --5b
  feature:add(cudnn.SpatialAveragePooling(7, 7, 1, 1, 0, 0))
  -- 1
  local classifier = nn.Sequential()
  classifier:add(nn.View(1024))
  classifier:add(nn.Linear(1024, opt.nClasses))
  classifier:add(cudnn.LogSoftMax())

  return feature, classifier
end

--[[
feature = inception_feature()
feature:cuda()
x = torch.Tensor(32, 3, 224, 224):uniform():cuda()
z = feature:forward(x)
print(z:size())
print(feature:backward(x, z):size())
--]]
