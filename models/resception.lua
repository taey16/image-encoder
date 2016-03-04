
require 'cudnn'
require 'cunn'
paths.dofile('inception_module.lua')


--  Rethinking the Inception Architecture for Computer Vision, arXiv, 2015
function createModel()
  -- 342
  local feature = nn.Sequential() 
  -- 299 valid
  feature:add(cudnn.SpatialConvolution(3, 32, 3, 3, 2, 2, 0, 0))
  feature:add(cudnn.SpatialBatchNormalization(32, std_epsilon, nil, true))
  feature:add(cudnn.ReLU(true))
  -- floor( 299 / 2) = 149
  local feature_main_flow = nn.Sequential()
  feature_main_flow:add(cudnn.SpatialConvolution(32,32, 3, 3, 1, 1, 0, 0))
  feature_main_flow:add(cudnn.SpatialBatchNormalization(32, std_epsilon, nil, true))
  feature_main_flow:add(cudnn.ReLU(true))
  -- 147
  feature_main_flow:add(cudnn.SpatialConvolution(32,64, 3, 3, 1, 1, 1, 1))
  feature_main_flow:add(cudnn.SpatialBatchNormalization(64, std_epsilon, nil, true))
  --feature_main_flow:add(cudnn.ReLU(true))
  -- 147
  local shortcut_flow = nn.Sequential()
  shortcut_flow:add(cudnn.SpatialConvolution(32,64, 3, 3, 1, 1, 0, 0))
  shortcut_flow:add(cudnn.SpatialBatchNormalization(64, std_epsilon, nil, true))
  local concat_flow = nn.ConcatTable()
  concat_flow:add(feature_main_flow)
  concat_flow:add(shortcut_flow)
  local add_concat_flow = nn.Sequential()
  add_concat_flow:add(concat_flow)
  add_concat_flow:add(nn.CAddTable())
  add_concat_flow:add(cudnn.ReLU(true))
  feature:add(add_concat_flow)
  -- 147
  feature:add(cudnn.SpatialMaxPooling(3, 3, 2, 2, 0, 0))
  -- floor( 147 / 2 ) = 73
  local feature_main_flow_1 = nn.Sequential()
  feature_main_flow_1:add(cudnn.SpatialConvolution(64, 80, 1, 1, 1, 1, 0, 0))
  feature_main_flow_1:add(cudnn.SpatialBatchNormalization(80, std_epsilon, nil, true))
  feature_main_flow_1:add(cudnn.ReLU(true))
  -- 73
  feature_main_flow_1:add(cudnn.SpatialConvolution(80, 192, 3, 3, 1, 1, 0, 0))
  feature_main_flow_1:add(cudnn.SpatialBatchNormalization(192, std_epsilon, nil, true))
  --feature_main_flow_1:add(cudnn.ReLU(true))
  -- 71
  local shortcut_flow_1 = nn.Sequential()
  shortcut_flow_1:add(cudnn.SpatialConvolution(64, 192, 3, 3, 1, 1, 0, 0))
  shortcut_flow_1:add(cudnn.SpatialBatchNormalization(192, std_epsilon, nil, true))
  local concat_flow_1 = nn.ConcatTable()
  concat_flow_1:add(feature_main_flow_1)
  concat_flow_1:add(shortcut_flow_1)
  local add_concat_flow_1 = nn.Sequential()
  add_concat_flow_1:add(concat_flow_1)
  add_concat_flow_1:add(nn.CAddTable())
  add_concat_flow_1:add(cudnn.ReLU(true))
  feature:add(add_concat_flow_1)
  -- 71
  feature:add(cudnn.SpatialMaxPooling(3, 3, 2, 2, 0, 0))
  -- floor( 71 / 2 ) = 35

  mixed_1 = resception_module( 1, 192, {{64 }, { 48,  64},{64, 96, 96}, {32}})
  -- 192 + 64 + 64 + 96 + 32 = 256 (similar to fig.4,5)
  mixed_2 = resception_module( 2, 256, {{64 }, { 48,  64},{64, 96, 96}, {64}})
  mixed_3 = resception_module( 3, 288, {{64 }, { 48,  64},{64, 96, 96}, {64}})
  -- similar to fig.10
  mixed_4 = resception_module( 4, 288, {{384}, { 64,  96,  96}})

  -- floor( 35 / 2 ) = 17
  -- 288 + 384 + 96 = 768 (fig.6)
  mixed_5 = resception_module( 5, 768, {{192}, {128, 128, 192}, {128, 128, 128, 128, 192}, {192}})
  mixed_6 = resception_module( 6, 768, {{192}, {160, 160, 192}, {160, 160, 160, 160, 192}, {192}})
  mixed_7 = resception_module( 7, 768, {{192}, {160, 160, 192}, {160, 160, 160, 160, 192}, {192}})
  mixed_8 = resception_module( 8, 768, {{192}, {192, 192, 192}, {192, 192, 192, 192, 192}, {192}})
  -- similar to fig.10
  mixed_9 = resception_module( 9, 768, {{192, 320}, {192, 192, 192, 192}})

  -- floor(17 / 2) = 8
  -- 768 + 320 + 192 = 1280 (fig.7)
  mixed_10= resception_module(10,1280, {{320}, {384, 384, 384}, {448, 384, 384, 384}, {192}})
  -- 320 + (384 + 384) + (384 + 384) + 192 = 2048  (fig.7)
  mixed_11= resception_module(11,2048, {{320}, {384, 384, 384}, {448, 384, 384, 384}, {192}})

  feature:add(mixed_1)
  feature:add(mixed_2)
  feature:add(mixed_3)
  feature:add(mixed_4)
  feature:add(mixed_5)
  feature:add(mixed_6)
  feature:add(mixed_7)
  feature:add(mixed_8)
  feature:add(mixed_9)
  feature:add(mixed_10)
  feature:add(mixed_11)
  feature:add(cudnn.SpatialAveragePooling(8, 8, 1, 1, 0, 0))

  feature:get(1).gradInput = nil

  feature:add(nn.View(2048))
  feature:add(nn.Linear(2048, opt.nClasses))
  feature:add(cudnn.LogSoftMax())

  --[[
  local classifier = nn.Sequential()
  classifier:add(nn.View(2048))
  classifier:add(nn.Linear(2048, opt.nClasses))
  classifier:add(cudnn.LogSoftMax())
  --]]

  --return feature, classifier
  return feature
end

--[[
feature = createModel()
feature:cuda()
x = torch.CudaTensor(32, 3, 299, 299):normal()
z = feature:forward(x)
print(z:size())
print(feature:backward(x, z):size())
--]]

