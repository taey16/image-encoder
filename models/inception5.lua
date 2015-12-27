
require 'cudnn'
require 'cunn'
paths.dofile('inception_module.lua')


function createModel()
  local feature = nn.Sequential() 
   
  -- 224
  feature:add(cudnn.SpatialConvolution(3, 64, 7, 7, 2, 2, 3, 3))
  feature:add(nn.SpatialBatchNormalization(64))
  feature:add(cudnn.ReLU(true))
  feature:add(cudnn.SpatialMaxPooling(2, 2, 2, 2, 0, 0))
  -- 56
  feature:add(cudnn.SpatialConvolution(64,128, 3, 3, 1, 1, 1, 1))
  feature:add(nn.SpatialBatchNormalization(128))
  feature:add(cudnn.ReLU(true))
  feature:add(cudnn.SpatialConvolution(128,128, 3, 3, 1, 1, 1, 1))
  feature:add(nn.SpatialBatchNormalization(128))
  feature:add(cudnn.ReLU(true))
  feature:add(cudnn.SpatialMaxPooling(2, 2, 2, 2, 0, 0))
  -- 28
  feature:add(inception_module(2, 128, {{64 }, {96,  128}, {16, 32}, {32}}))
  feature:add(inception_module(2, 256, {{128}, {128, 192}, {32, 96}, {64}}))
  feature:add(nn.SpatialMaxPooling(3, 3, 2, 2, 1, 1))
  -- 14
  feature:add(inception_module(2, 480, {{192}, {96,  208}, {16, 48}, {64}}))
  feature:add(inception_module(2, 512, {{160}, {112, 224}, {24, 64}, {64}}))
  feature:add(inception_module(2, 512, {{128}, {128, 256}, {24, 64}, {64}}))
  feature:add(inception_module(2, 512, {{112}, {144, 288}, {32, 64}, {64}}))
  feature:add(inception_module(2, 528, {{256}, {160, 320}, {32, 128}, {128}}))
  feature:add(nn.SpatialMaxPooling(3, 3, 2, 2, 1, 1))
  -- 7
  feature:add(inception_module(2, 832, {{256}, {160, 320}, {32, 128}, {128}}))
  feature:add(inception_module(2, 832, {{384}, {192, 384}, {48, 128}, {128}}))
  -- global avgpool
  feature:add(cudnn.SpatialAveragePooling(7, 7, 1, 1, 0, 0))

  local classifier = nn.Sequential()
  classifier:add(nn.View(1*1*1024))
  classifier:add(nn.Linear(1*1*1024, 1000))
  --classifier:add(nn.Linear(1*1*1024, 10))
  classifier:add(cudnn.LogSoftMax())

  --[[
  feature:add(nn.View(1*1*1024))
  --classifier:add(nn.Linear(1*1*1024, 1000))
  feature:add(nn.Linear(1*1*1024, 10))
  feature:add(cudnn.LogSoftMax())
  --]]

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
