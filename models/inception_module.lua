
require 'cudnn'
require 'cunn'


function inception7_residual_module(depth_dim, input_size, stride, config)
  local conv1 = nil   
  local conv3 = nil
  local double_conv3 = nil
  local pool = nil

  local conv3_concat = nil
  local conv3_shortcut = nil
  local conv3_add = nil
  local double_conv3_concat = nil
  local double_conv3_shortcut = nil
  local double_conv3_add = nil
   
  local depth_concat = nn.DepthConcat(depth_dim)

  if config[1][1] > 0 then
    conv1 = nn.Sequential()
    conv1:add(cudnn.SpatialConvolution(input_size, config[1][1], 1, 1, 1, 1, 0, 0))
    conv1:add(nn.SpatialBatchNormalization(config[1][1]))
    conv1:add(cudnn.ReLU(true))
    depth_concat:add(conv1)
  end

  conv3 = nn.Sequential()
  conv3:add(cudnn.SpatialConvolution(input_size, config[2][1], 1, 1, 1, 1, 0, 0))
  conv3:add(nn.SpatialBatchNormalization(config[2][1]))
  conv3:add(cudnn.ReLU(true))
  conv3:add(cudnn.SpatialConvolution(config[2][1], config[2][2], 1, 3, stride, stride, 0, 1))
  conv3:add(nn.SpatialBatchNormalization(config[2][2]))
  conv3:add(cudnn.ReLU(true))
  conv3:add(cudnn.SpatialConvolution(config[2][2], config[2][2], 3, 1, stride, stride, 1, 0))
  conv3:add(nn.SpatialBatchNormalization(config[2][2]))
  --conv3:add(cudnn.ReLU(true))
  if stride == 1 then
    conv3_add = nn.Sequential()
    conv3_concat = nn.ConcatTable()
    conv3_shortcut = nn.Sequential()
    conv3_shortcut:add(cudnn.SpatialConvolution(input_size, config[2][2], 1, 1, stride, stride, 0, 0))
    conv3_shortcut:add(nn.SpatialBatchNormalization(config[2][2]))
    conv3_concat:add(conv3)
    conv3_concat:add(conv3_shortcut)
    conv3_add:add(conv3_concat)
    conv3_add:add(nn.CAddTable())
    conv3_add:add(cudnn.ReLU(true))
    depth_concat:add(conv3_add)
  else
    conv3:add(cudnn.ReLU(true))
    depth_concat:add(conv3)
  end

  double_conv3 = nn.Sequential()
  double_conv3:add(cudnn.SpatialConvolution(input_size, config[3][1], 1, 1, 1, 1, 0, 0))
  double_conv3:add(nn.SpatialBatchNormalization(config[3][1]))
  double_conv3:add(cudnn.ReLU(true))
  double_conv3:add(cudnn.SpatialConvolution(config[3][1], config[3][2], 1, 3, 1, 1, 0, 1))
  double_conv3:add(nn.SpatialBatchNormalization(config[3][2]))
  double_conv3:add(cudnn.ReLU(true))
  double_conv3:add(cudnn.SpatialConvolution(config[3][2], config[3][2], 3, 1, 1, 1, 1, 0))
  double_conv3:add(nn.SpatialBatchNormalization(config[3][2]))
  double_conv3:add(cudnn.ReLU(true))
  double_conv3:add(cudnn.SpatialConvolution(config[3][2], config[3][2], 1, 3, stride, stride, 0, 1))
  double_conv3:add(nn.SpatialBatchNormalization(config[3][2]))
  double_conv3:add(cudnn.ReLU(true))
  double_conv3:add(cudnn.SpatialConvolution(config[3][2], config[3][2], 3, 1, stride, stride, 1, 0))
  double_conv3:add(nn.SpatialBatchNormalization(config[3][2]))
  --double_conv3:add(cudnn.ReLU(true))
  if stride == 1 then
    double_conv3_add = nn.Sequential()
    double_conv3_concat = nn.ConcatTable()
    double_conv3_shortcut = nn.Sequential()
    double_conv3_shortcut:add(cudnn.SpatialConvolution(input_size, config[3][2], 1, 1, stride, stride, 0, 0))
    double_conv3_shortcut:add(nn.SpatialBatchNormalization(config[3][2]))
    double_conv3_concat:add(double_conv3)
    double_conv3_concat:add(double_conv3_shortcut)
    double_conv3_add:add(double_conv3_concat)
    double_conv3_add:add(nn.CAddTable())
    double_conv3_add:add(cudnn.ReLU(true))
    depth_concat:add(double_conv3_add)
  else
    double_conv3:add(cudnn.ReLU(true))
    depth_concat:add(double_conv3)
  end

  pool = nn.Sequential()
  if config[4][1] == 'avg' then
    pool:add(cudnn.SpatialAveragePooling(3, 3, stride, stride, 1, 1))
  else
    pool:add(cudnn.SpatialMaxPooling(3, 3, stride, stride, 1, 1))
  end
  if config[4][2] > 0 then
    pool:add(cudnn.SpatialConvolution(input_size, config[4][2], 1, 1, 1, 1, 0, 0))
    pool:add(nn.SpatialBatchNormalization(config[4][2]))
    pool:add(cudnn.ReLU(true))
  end
  depth_concat:add(pool)
  
  return depth_concat
end


function inception7_module(depth_dim, input_size, stride, config)
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
  conv3:add(cudnn.SpatialConvolution(input_size, config[2][1], 1, 1, 1, 1, 0, 0))
  conv3:add(nn.SpatialBatchNormalization(config[2][1]))
  conv3:add(cudnn.ReLU(true))
  conv3:add(cudnn.SpatialConvolution(config[2][1], config[2][2], 1, 3, stride, stride, 0, 1))
  conv3:add(nn.SpatialBatchNormalization(config[2][2]))
  conv3:add(cudnn.ReLU(true))
  conv3:add(cudnn.SpatialConvolution(config[2][2], config[2][2], 3, 1, stride, stride, 1, 0))
  conv3:add(nn.SpatialBatchNormalization(config[2][2]))
  conv3:add(cudnn.ReLU(true))
  depth_concat:add(conv3)

  double_conv3 = nn.Sequential()
  double_conv3:add(cudnn.SpatialConvolution(input_size, config[3][1], 1, 1, 1, 1, 0, 0))
  double_conv3:add(nn.SpatialBatchNormalization(config[3][1]))
  double_conv3:add(cudnn.ReLU(true))
  double_conv3:add(cudnn.SpatialConvolution(config[3][1], config[3][2], 1, 3, 1, 1, 0, 1))
  double_conv3:add(nn.SpatialBatchNormalization(config[3][2]))
  double_conv3:add(cudnn.ReLU(true))
  double_conv3:add(cudnn.SpatialConvolution(config[3][2], config[3][2], 3, 1, 1, 1, 1, 0))
  double_conv3:add(nn.SpatialBatchNormalization(config[3][2]))
  double_conv3:add(cudnn.ReLU(true))
  double_conv3:add(cudnn.SpatialConvolution(config[3][2], config[3][2], 1, 3, stride, stride, 0, 1))
  double_conv3:add(nn.SpatialBatchNormalization(config[3][2]))
  double_conv3:add(cudnn.ReLU(true))
  double_conv3:add(cudnn.SpatialConvolution(config[3][2], config[3][2], 3, 1, stride, stride, 1, 0))
  double_conv3:add(nn.SpatialBatchNormalization(config[3][2]))
  double_conv3:add(cudnn.ReLU(true))
  depth_concat:add(double_conv3)

  pool = nn.Sequential()
  if config[4][1] == 'avg' then
    pool:add(cudnn.SpatialAveragePooling(3, 3, stride, stride, 1, 1))
  else
    pool:add(cudnn.SpatialMaxPooling(3, 3, stride, stride, 1, 1))
  end
  if config[4][2] > 0 then
    pool:add(cudnn.SpatialConvolution(input_size, config[4][2], 1, 1, 1, 1, 0, 0))
    pool:add(nn.SpatialBatchNormalization(config[4][2]))
    pool:add(cudnn.ReLU(true))
  end
  depth_concat:add(pool)
  
  return depth_concat
end


function inception6_module(depth_dim, input_size, stride, config)
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
  conv3:add(cudnn.SpatialConvolution(input_size, config[2][1], 1, 1, 1, 1, 0, 0))
  conv3:add(nn.SpatialBatchNormalization(config[2][1]))
  conv3:add(cudnn.ReLU(true))
  conv3:add(cudnn.SpatialConvolution(config[2][1], config[2][2], 3, 3, stride, stride, 1, 1))
  conv3:add(nn.SpatialBatchNormalization(config[2][2]))
  conv3:add(cudnn.ReLU(true))
  depth_concat:add(conv3)

  double_conv3 = nn.Sequential()
  double_conv3:add(cudnn.SpatialConvolution(input_size, config[3][1], 1, 1, 1, 1, 0, 0))
  double_conv3:add(nn.SpatialBatchNormalization(config[3][1]))
  double_conv3:add(cudnn.ReLU(true))
  double_conv3:add(cudnn.SpatialConvolution(config[3][1], config[3][2], 3, 3, 1, 1, 1, 1))
  double_conv3:add(nn.SpatialBatchNormalization(config[3][2]))
  double_conv3:add(cudnn.ReLU(true))
  double_conv3:add(cudnn.SpatialConvolution(config[3][2], config[3][2], 3, 3, stride, stride, 1, 1))
  double_conv3:add(nn.SpatialBatchNormalization(config[3][2]))
  double_conv3:add(cudnn.ReLU(true))
  depth_concat:add(double_conv3)

  pool = nn.Sequential()
  if config[4][1] == 'avg' then
    pool:add(cudnn.SpatialAveragePooling(3, 3, stride, stride, 1, 1))
  else
    pool:add(cudnn.SpatialMaxPooling(3, 3, stride, stride, 1, 1))
  end
  if config[4][2] > 0 then
    pool:add(cudnn.SpatialConvolution(input_size, config[4][2], 1, 1, 1, 1, 0, 0))
    pool:add(nn.SpatialBatchNormalization(config[4][2]))
    pool:add(cudnn.ReLU(true))
  end
  depth_concat:add(pool)
  
  return depth_concat
end

function inception_module(depth_dim, input_size, stride, config)
  local conv1 = nil
  local conv3 = nil
  local conv5 = nil
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

  conv5 = nn.Sequential()
  conv5:add(cudnn.SpatialConvolution(input_size, config[3][1], 1, 1, stride, stride, 0, 0))
  conv5:add(nn.SpatialBatchNormalization(config[3][1]))
  conv5:add(cudnn.ReLU(true))
  conv5:add(cudnn.SpatialConvolution(config[3][1], config[3][2], 5, 5, stride, stride, 1, 1))
  conv5:add(nn.SpatialBatchNormalization(config[3][2]))
  conv5:add(cudnn.ReLU(true))
  depth_concat:add(conv5)

  pool = nn.Sequential()
  if config[4][1] == 'avg' then
    pool:add(cudnn.SpatialAveragePooling(3, 3, stride, stride, 1, 1))
  else
    pool:add(cudnn.SpatialMaxPooling(3, 3, stride, stride, 1, 1))
  end
  if config[4][2] > 0 then
    pool:add(cudnn.SpatialConvolution(input_size, config[4][2], 1, 1, 1, 1, 0, 0))
    pool:add(nn.SpatialBatchNormalization(config[4][2]))
    pool:add(cudnn.ReLU(true))
  end
  depth_concat:add(pool)
  
  return depth_concat
end

