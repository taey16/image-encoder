
local MaxPooling
local AvgPooling
local SpatialConv 
local ReLU
local LogSoftMax

if opt.backend == 'nn' then
  MaxPooling = nn.SpatialMaxPooling
  AvgPooling = nn.SpatialAvgPooling
  SpatialConv= nn.SpatialConvolution
  ReLU = nn.ReLU
  LogSoftMax = nn.LogSoftMax
elseif opt.backend == 'cudnn' then
  MaxPooling = cudnn.SpatialMaxPooling
  AvgPooling = cudnn.SpatialAvgPooling
  SpatialConv= cudnn.SpatialConvolution
  ReLU = cudnn.ReLU
  LogSoftMax = cudnn.LogSoftMax
else
  error('Check backend '..opt.backend)
end
print('===> backend: ' .. opt.backend)

local model = nn.Sequential()

function createModel()
  -- 32, 24, 16
  model:add(SpatialConv(1, 16, 3, 3, 1, 1, 1, 1))
  model:add(nn.SpatialBatchNormalization(16))
  model:add(ReLU(true))
  model:add(nn.Dropout(0.3))
  model:add(SpatialConv(16, 16, 3, 3, 1, 1, 1, 1))
  model:add(nn.SpatialBatchNormalization(16))
  model:add(ReLU(true))
  model:add(nn.Dropout(0.3))
  model:add(SpatialConv(16, 32, 1, 1, 1, 1, 0, 0))
  model:add(nn.SpatialBatchNormalization(32))
  model:add(ReLU(true))
  model:add(SpatialConv(32, 48, 1, 1, 1, 1, 0, 0))
  model:add(nn.SpatialBatchNormalization(48))
  model:add(ReLU(true))
  model:add(MaxPooling(2, 2, 2, 2, 0, 0))
  -- 16, 12, 8
  model:add(SpatialConv(48, 48, 3, 3, 1, 1, 1, 1))
  model:add(nn.SpatialBatchNormalization(48))
  model:add(ReLU(true))
  model:add(nn.Dropout(0.4))
  model:add(SpatialConv(48, 48, 3, 3, 1, 1, 1, 1))
  model:add(nn.SpatialBatchNormalization(48))
  model:add(ReLU(true))
  model:add(nn.Dropout(0.4))
  model:add(SpatialConv(48, 64, 1, 1, 1, 1, 0, 0))
  model:add(nn.SpatialBatchNormalization(64))
  model:add(ReLU(true))
  model:add(SpatialConv(64, 64, 1, 1, 1, 1, 0, 0))
  model:add(nn.SpatialBatchNormalization(64))
  model:add(ReLU(true))
  model:add(MaxPooling(2, 2, 2, 2, 0, 0))
  -- 8, 6, 4
  local resolution = opt.sampling_grid_size/2/2

  model:add(nn.View(resolution*resolution*64))
  model:add(nn.Linear(resolution*resolution*64, 320))
  model:add(nn.BatchNormalization(320))
  model:add(ReLU(true))
  model:add(nn.Dropout(0.5))
  model:add(nn.Linear(320, 10))
  model:add(LogSoftMax())

  return model

end

