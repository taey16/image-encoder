
require 'cudnn'
require 'cunn'
paths.dofile('inception_module.lua')


function createModel()
  -- 256, 292
  local feature = nn.Sequential() 
  -- 224, 256
  feature:add(cudnn.SpatialConvolution(3, 64, 7, 7, 2, 2, 3, 3))
  --    , 128
  feature:add(nn.SpatialBatchNormalization(64))
  feature:add(cudnn.ReLU(true))
  feature:add(cudnn.SpatialMaxPooling(3, 3, 2, 2, 1, 1))
  -- 56, 64
  feature_main_flow = nn.Sequential()
  feature_main_flow:add(cudnn.SpatialConvolution(64,64, 3, 3, 1, 1, 1, 1))
  feature_main_flow:add(nn.SpatialBatchNormalization(64))
  feature_main_flow:add(cudnn.ReLU(true))
  feature_main_flow:add(cudnn.SpatialConvolution(64,192, 3, 3, 1, 1, 1, 1))
  feature_main_flow:add(nn.SpatialBatchNormalization(192))
  -- start to projection shortcuts
  feature_concat = nn.ConcatTable()
  feature_shortcut = nn.Sequential()
  feature_shortcut:add(cudnn.SpatialConvolution(64,192,1,1,1,1,0,0))
  feature_shortcut:add(nn.SpatialBatchNormalization(192))
  feature_concat:add(feature_main_flow)
  feature_concat:add(feature_shortcut)
  feature:add(feature_concat)
  feature:add(nn.CAddTable())
  feature:add(cudnn.ReLU(true))
  feature:add(cudnn.SpatialMaxPooling(3, 3, 2, 2, 1, 1))
  -- 28, 32
  feature:add(inception7_residual_module(2, 192, 1, {{ 64}, {64,  64}, {64,  96}, {'avg', 32}}))
  feature:add(inception7_residual_module(2, 256, 1, {{ 64}, {64,  96}, {64,  96}, {'avg', 64}}))
  feature:add(inception7_residual_module(2, 320, 2, {{  0}, {128,160}, {64,  96}, {'max',  0}}))
  -- 14, 16
  feature:add(inception7_residual_module(2, 576, 1, {{224}, {64,  96}, {96, 128}, {'avg',128}}))
  feature:add(inception7_residual_module(2, 576, 1, {{192}, {96 ,128}, {96, 128}, {'avg',128}}))
  feature:add(inception7_residual_module(2, 576, 1, {{160}, {128,160}, {128,128}, {'avg',128}}))
  feature:add(inception7_residual_module(2, 576, 1, {{ 96}, {128,192}, {160,160}, {'avg',128}}))
  feature:add(inception7_residual_module(2, 576, 1, {{ 96}, {128,192}, {160,160}, {'avg',128}}))
  feature:add(inception7_residual_module(2, 576, 2, {{  0}, {128,192}, {192,256}, {'max',  0}}))
  -- 7, 8
  feature:add(inception7_residual_module(2,1024, 1, {{352}, {192,320}, {160,224}, {'avg',128}}))
  feature:add(inception7_residual_module(2,1024, 1, {{352}, {192,320}, {192,224}, {'avg',128}}))
  feature:add(inception7_residual_module(2,1024, 2, {{  0}, {192,608}, {192,416}, {'max',  0}}))
  --    4
  feature:add(inception7_residual_module(2,2048, 1, {{704}, {256,640}, {256,448}, {'avg',256}}))
  feature:add(cudnn.SpatialAveragePooling(4, 4, 1, 1, 0, 0))
  -- 1
  local classifier = nn.Sequential()
  classifier:add(nn.View(2048))
  classifier:add(nn.Linear(2048, opt.nClasses))
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
