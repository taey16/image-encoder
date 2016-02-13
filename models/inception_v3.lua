
require 'cudnn'
require 'cunn'
paths.dofile('inception_module.lua')


function createModel()
  -- 342
  local feature = nn.Sequential() 
  -- 299 valid
  feature:add(cudnn.SpatialConvolution(3, 32, 3, 3, 2, 2, 0, 0))
  feature:add(nn.SpatialBatchNormalization(32, std_epsilon, nil, true))
  feature:add(cudnn.ReLU(true))
  -- (299 - 2) / 2 = 149
  feature:add(cudnn.SpatialConvolution(32,32, 3, 3, 1, 1, 0, 0))
  feature:add(nn.SpatialBatchNormalization(32, std_epsilon, nil, true))
  feature:add(cudnn.ReLU(true))
  -- 147
  feature:add(cudnn.SpatialConvolution(32,64, 3, 3, 1, 1, 1, 1))
  feature:add(nn.SpatialBatchNormalization(64, std_epsilon, nil, true))
  feature:add(cudnn.ReLU(true))
  -- 147
  feature:add(cudnn.SpatialMaxPooling(3, 3, 2, 2, 0, 0))
  feature:add(cudnn.SpatialConvolution(64, 80, 1, 1, 1, 1, 0, 0))
  feature:add(nn.SpatialBatchNormalization(80, std_epsilon, nil, true))
  feature:add(cudnn.ReLU(true))
  feature:add(cudnn.SpatialConvolution(80, 192, 3, 3, 1, 1, 0, 0))
  feature:add(nn.SpatialBatchNormalization(192, std_epsilon, nil, true))
  feature:add(cudnn.ReLU(true))
  feature:add(cudnn.SpatialMaxPooling(3, 3, 2, 2, 0, 0))


  mixed_1 = inception_v3_module( 1, 192, {{64 }, { 48,  64},{64, 96, 96}, {32}})
  -- 192 + 64 + 64 + 96 + 32 = 256
  mixed_2 = inception_v3_module( 2, 256, {{64 }, { 48,  64},{64, 96, 96}, {64}})
  mixed_3 = inception_v3_module( 3, 288, {{64 }, { 48,  64},{64, 96, 96}, {64}})
  mixed_4 = inception_v3_module( 4, 288, {{384}, { 64,  96,  96}})
  -- 288 + 384 + 96 = 768
  mixed_5 = inception_v3_module( 5, 768, {{192}, {128, 128, 192}, {128, 128, 128, 128, 192}, {192}})
  mixed_6 = inception_v3_module( 6, 768, {{192}, {160, 160, 192}, {160, 160, 160, 160, 192}, {192}})
  mixed_7 = inception_v3_module( 7, 768, {{192}, {160, 160, 192}, {160, 160, 160, 160, 192}, {192}})
  mixed_8 = inception_v3_module( 8, 768, {{192}, {192, 192, 192}, {192, 192, 192, 192, 192}, {192}})
  mixed_9 = inception_v3_module( 9, 768, {{192, 320}, {192, 192, 192, 192}})
  -- 768 + 320 + 192 = 1280
  mixed_10= inception_v3_module(10,1280, {{320}, {384, 384, 384}, {448, 384, 384, 384}, {192}})
  -- 320 + (384 + 384) + (384 + 384) + 192 = 2048 
  mixed_11= inception_v3_module(11,2048, {{320}, {384, 384, 384}, {448, 384, 384, 384}, {192}})

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

  local classifier = nn.Sequential()
  classifier:add(nn.View(2048))
  classifier:add(nn.Linear(2048, opt.nClasses))
  classifier:add(cudnn.LogSoftMax())

  return feature, classifier
end

--[[
feature = createModel()
feature:cuda()
x = torch.CudaTensor(32, 3, 299, 299):normal()
z = feature:forward(x)
print(z:size())
print(feature:backward(x, z):size())
--]]

