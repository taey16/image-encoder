
require 'cudnn'
require 'cunn'
paths.dofile('init_model_weight.lua')

function inception_module(depth_dim, input_size, config)
  local conv1 = nil   
  local conv3 = nil
  local conv5 = nil
  local pool = nil
   
  local depth_concat = nn.DepthConcat(depth_dim)
  conv1 = nn.Sequential()
  conv1:add(cudnn.SpatialConvolution(input_size, config[1][1], 1, 1, 1, 1, 0, 0))
  conv1:add(nn.SpatialBatchNormalization(config[1][1]))
  conv1:add(cudnn.ReLU(true))

  depth_concat:add(conv1)

  conv3 = nn.Sequential()
  conv3:add(cudnn.SpatialConvolution(input_size, config[2][1], 1, 1, 1, 1, 0, 0))
  conv3:add(nn.SpatialBatchNormalization(config[2][1]))
  conv3:add(cudnn.ReLU(true))
  conv3:add(cudnn.SpatialConvolution(config[2][1], config[2][2], 3, 3, 1, 1, 1, 1))
  conv3:add(nn.SpatialBatchNormalization(config[2][2]))
  conv3:add(cudnn.ReLU(true))

  depth_concat:add(conv3)

  conv5 = nn.Sequential()
  conv5:add(cudnn.SpatialConvolution(input_size, config[3][1], 1, 1, 1, 1, 0, 0))
  conv5:add(nn.SpatialBatchNormalization(config[3][1]))
  conv5:add(cudnn.ReLU(true))
  conv5:add(cudnn.SpatialConvolution(config[3][1], config[3][2], 3, 3, 1, 1, 1, 1))
  conv5:add(nn.SpatialBatchNormalization(config[3][2]))
  conv5:add(cudnn.ReLU(true))
  conv5:add(cudnn.SpatialConvolution(config[3][2], config[3][2], 3, 3, 1, 1, 1, 1))
  conv5:add(nn.SpatialBatchNormalization(config[3][2]))
  conv5:add(cudnn.ReLU(true))

  depth_concat:add(conv5)

  pool = nn.Sequential()
  pool:add(nn.SpatialMaxPooling(3, 3, 1, 1, 1, 1))
  pool:add(cudnn.SpatialConvolution(input_size, config[4][1], 1, 1, 1, 1, 0, 0))
  pool:add(nn.SpatialBatchNormalization(config[4][1]))
  pool:add(cudnn.ReLU(true))

  depth_concat:add(pool)
  
  return depth_concat
end

function createModel(nGPU) -- validate.lua Acc:
  local model = nn.Sequential() 
   
  -- 224
  model:add(cudnn.SpatialConvolution(3, 64, 7, 7, 2, 2, 3, 3))
  model:add(nn.SpatialBatchNormalization(64))
  model:add(cudnn.ReLU(true))
  model:add(cudnn.SpatialMaxPooling(2, 2, 2, 2, 0, 0))
  -- 56
  model:add(cudnn.SpatialConvolution(64,128, 3, 3, 1, 1, 1, 1))
  model:add(nn.SpatialBatchNormalization(128))
  model:add(cudnn.ReLU(true))
  model:add(cudnn.SpatialConvolution(128,128, 3, 3, 1, 1, 1, 1))
  model:add(nn.SpatialBatchNormalization(128))
  model:add(cudnn.ReLU(true))
  model:add(cudnn.SpatialMaxPooling(2, 2, 2, 2, 0, 0))
  -- 28
  model:add(inception_module(2, 128, {{64 }, {96,  128}, {16, 32}, {32}}))
  model:add(inception_module(2, 256, {{128}, {128, 192}, {32, 96}, {64}}))
  model:add(nn.SpatialMaxPooling(3, 3, 2, 2, 1, 1))
  -- 14
  model:add(inception_module(2, 480, {{192}, {96,  208}, {16, 48}, {64}}))
  model:add(inception_module(2, 512, {{160}, {112, 224}, {24, 64}, {64}}))
  model:add(inception_module(2, 512, {{128}, {128, 256}, {24, 64}, {64}}))
  model:add(inception_module(2, 512, {{112}, {144, 288}, {32, 64}, {64}}))
  model:add(inception_module(2, 528, {{256}, {160, 320}, {32, 128}, {128}}))
  model:add(nn.SpatialMaxPooling(3, 3, 2, 2, 1, 1))
  -- 7
  model:add(inception_module(2, 832, {{256}, {160, 320}, {32, 128}, {128}}))
  model:add(inception_module(2, 832, {{384}, {192, 384}, {48, 128}, {128}}))
  -- global avgpool
  model:add(cudnn.SpatialAveragePooling(7, 7, 1, 1, 0, 0))
  model:add(nn.View(1*1*1024))
  model:add(nn.Linear(1*1*1024, 1000))
  model:add(cudnn.LogSoftMax())

  MSRinit( model )

  return model
end

--[[
model = inception_model()
model:cuda()
x = torch.Tensor(32, 3, 224, 224):uniform():cuda()
z = model:forward(x)
print(z:size())
print(model:backward(x, z):size())
--]]
