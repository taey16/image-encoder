
require 'cudnn'
require 'cunn'
paths.dofile('inception_module.lua')


function createModel()
  -- 256, 292
  local feature = nn.Sequential() 
  -- 224, 256
  feature:add(cudnn.SpatialConvolution(3, 64, 7, 7, 2, 2, 3, 3))
  feature:add(nn.SpatialBatchNormalization(64))
  feature:add(cudnn.ReLU(true))
  feature:add(cudnn.SpatialMaxPooling(3, 3, 2, 2, 1, 1))
  -- 56, 64
  feature:add(cudnn.SpatialConvolution(64,64, 3, 3, 1, 1, 1, 1))
  feature:add(nn.SpatialBatchNormalization(64))
  feature:add(cudnn.ReLU(true))
  feature:add(cudnn.SpatialConvolution(64,192, 3, 3, 1, 1, 1, 1))
  feature:add(nn.SpatialBatchNormalization(192))
  feature:add(cudnn.ReLU(true))
  feature:add(cudnn.SpatialMaxPooling(3, 3, 2, 2, 1, 1))
  -- 28, 32
  feature:add(inception7_module(2, 192, 1, {{ 64}, {64,  64}, {64,  96}, {'avg', 32}}))
  feature:add(inception7_module(2, 256, 1, {{ 64}, {64,  96}, {64,  96}, {'avg', 64}}))
  feature:add(inception7_module(2, 320, 2, {{  0}, {128,160}, {64,  96}, {'max',  0}}))
  -- 14, 16
  feature:add(inception7_module(2, 576, 1, {{224}, {64,  96}, {96, 128}, {'avg',128}}))
  feature:add(inception7_module(2, 576, 1, {{192}, {96 ,128}, {96, 128}, {'avg',128}}))
  feature:add(inception7_module(2, 576, 1, {{160}, {128,160}, {128,128}, {'avg',128}}))
  feature:add(inception7_module(2, 576, 1, {{ 96}, {128,192}, {160,160}, {'avg',128}}))
  feature:add(inception7_module(2, 576, 2, {{  0}, {128,192}, {192,256}, {'max',  0}}))
  -- 7, 8
  feature:add(inception7_module(2,1024, 1, {{352}, {192,320}, {160,224}, {'avg',128}}))
  feature:add(inception7_module(2,1024, 1, {{352}, {192,320}, {192,224}, {'avg',128}}))
  feature:add(cudnn.SpatialAveragePooling(8, 8, 1, 1, 0, 0))
  -- 1
  local classifier = nn.Sequential()
  classifier:add(nn.View(1*1*1024))
  classifier:add(nn.Linear(1*1*1024, 1000))
  classifier:add(cudnn.LogSoftMax())

  return feature, classifier
  --return feature
end

--[[
feature = inception_feature()
feature:cuda()
x = torch.Tensor(32, 3, 224, 224):uniform():cuda()
z = feature:forward(x)
print(z:size())
print(feature:backward(x, z):size())
--]]
