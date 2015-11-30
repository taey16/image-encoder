
require 'cudnn'
require 'cunn'

function inception_module(depth_dim, input_size, stride, config)
  local conv1 = nil   
  local conv3 = nil
  local double_conv3 = nil
  local pool = nil
   
  local depth_concat = nn.DepthConcat(depth_dim)

  if config[1][1] > 0 then
    conv1 = nn.Sequential()
    conv1:add(cudnn.SpatialConvolution(input_size, config[1][1], 1, 1, 1, 1, 0, 0))
    conv1:add(nn.SpatialBatchNormalization(config[1][1]))
    conv1:add(cudnn.ReLU(true))
    depth_concat:add(conv1)
  end

  conv3 = nn.Sequential()
  conv3:add(cudnn.SpatialConvolution(input_size, config[2][1], 1, 1, stride, stride, 0, 0))
  conv3:add(nn.SpatialBatchNormalization(config[2][1]))
  conv3:add(cudnn.ReLU(true))
  conv3:add(cudnn.SpatialConvolution(config[2][1], config[2][2], 3, 3, stride, stride, 1, 1))
  conv3:add(nn.SpatialBatchNormalization(config[2][2]))
  conv3:add(cudnn.ReLU(true))
  depth_concat:add(conv3)

  double_conv3 = nn.Sequential()
  double_conv3:add(cudnn.SpatialConvolution(input_size, config[3][1], 1, 1, stride, stride, 0, 0))
  double_conv3:add(nn.SpatialBatchNormalization(config[3][1]))
  double_conv3:add(cudnn.ReLU(true))
  double_conv3:add(cudnn.SpatialConvolution(config[3][1], config[3][2], 3, 3, stride, stride, 1, 1))
  double_conv3:add(nn.SpatialBatchNormalization(config[3][2]))
  double_conv3:add(cudnn.ReLU(true))
  double_conv3:add(cudnn.SpatialConvolution(config[3][2], config[3][2], 3, 3, stride, stride, 1, 1))
  double_conv3:add(nn.SpatialBatchNormalization(config[3][2]))
  double_conv3:add(cudnn.ReLU(true))
  depth_concat:add(double_conv3)

  pool = nn.Sequential()
  if config[4][1] == 'avg' then
    pool:add(nn.SpatialAveragePooling(3, 3, stride, stride, 1, 1))
  else
    pool:add(nn.SpatialMaxPooling(3, 3, stride, stride, 1, 1))
  end
  if config[4][2] > 0 then
    pool:add(cudnn.SpatialConvolution(input_size, config[4][2], 1, 1, 1, 1, 0, 0))
    pool:add(nn.SpatialBatchNormalization(config[4][2]))
    pool:add(cudnn.ReLU(true))
  end
  depth_concat:add(pool)
  
  return depth_concat
end


function createModel()
  local feature = nn.Sequential() 
   
  -- 224
  feature:add(cudnn.SpatialConvolution(3, 64, 7, 7, 2, 2, 3, 3))
  feature:add(nn.SpatialBatchNormalization(64))
  feature:add(cudnn.ReLU(true))
  --feature:add(cudnn.SpatialMaxPooling(2, 2, 2, 2, 0, 0))
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
  feature:add(inception_module(2, 192, 1, {{ 64}, {64,  64}, {64,  96}, {'avg', 32}}))
  feature:add(inception_module(2, 256, 1, {{ 64}, {64,  96}, {64,  96}, {'avg', 64}}))
  feature:add(inception_module(2, 320, 2, {{  0}, {128,160}, {64,  96}, {'max',  0}}))
  -- 14
  feature:add(inception_module(2, 576, 1, {{224}, {64,  96}, {96, 128}, {'avg',128}}))
  feature:add(inception_module(2, 576, 1, {{192}, {96 ,128}, {96, 128}, {'avg',128}}))
  feature:add(inception_module(2, 576, 1, {{160}, {128,160}, {128,128}, {'avg',128}}))
  feature:add(inception_module(2, 576, 1, {{ 96}, {128,192}, {160,160}, {'avg',128}}))
  feature:add(inception_module(2, 576, 2, {{  0}, {128,192}, {192,256}, {'max',  0}}))
  -- 7
  feature:add(inception_module(2,1024, 1, {{352}, {192,320}, {160,224}, {'avg',128}}))
  feature:add(inception_module(2,1024, 1, {{352}, {192,320}, {192,224}, {'avg',128}}))
  feature:add(cudnn.SpatialAveragePooling(7, 7, 1, 1, 0, 0))
  -- 1
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
