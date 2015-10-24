
require 'cudnn'
require 'cunn'

local MaxPooling = cudnn.SpatialMaxPooling

function createModel(nGPU)
  local model = nn.Sequential()
  -- 32, 24, 16
  model:add(cudnn.SpatialConvolution(3, 16, 3, 3, 1, 1, 1, 1))
  model:add(nn.SpatialBatchNormalization(16))
  model:add(cudnn.ReLU(true))
  model:add(nn.Dropout(0.3))
  model:add(cudnn.SpatialConvolution(16, 16, 3, 3, 1, 1, 1, 1))
  model:add(nn.SpatialBatchNormalization(16))
  model:add(cudnn.ReLU(true))
  model:add(nn.Dropout(0.3))
  model:add(cudnn.SpatialConvolution(16, 32, 1, 1, 1, 1, 0, 0))
  model:add(nn.SpatialBatchNormalization(32))
  model:add(cudnn.ReLU(true))
  model:add(cudnn.SpatialConvolution(32, 48, 1, 1, 1, 1, 0, 0))
  model:add(nn.SpatialBatchNormalization(48))
  model:add(cudnn.ReLU(true))
  model:add(MaxPooling(2, 2, 2, 2, 0, 0))
  -- 16, 12, 8
  model:add(cudnn.SpatialConvolution(48, 48, 3, 3, 1, 1, 1, 1))
  model:add(nn.SpatialBatchNormalization(48))
  model:add(cudnn.ReLU(true))
  model:add(nn.Dropout(0.4))
  model:add(cudnn.SpatialConvolution(48, 48, 3, 3, 1, 1, 1, 1))
  model:add(nn.SpatialBatchNormalization(48))
  model:add(cudnn.ReLU(true))
  model:add(nn.Dropout(0.4))
  model:add(cudnn.SpatialConvolution(48, 64, 1, 1, 1, 1, 0, 0))
  model:add(nn.SpatialBatchNormalization(64))
  model:add(cudnn.ReLU(true))
  model:add(cudnn.SpatialConvolution(64, 64, 1, 1, 1, 1, 0, 0))
  model:add(nn.SpatialBatchNormalization(64))
  model:add(cudnn.ReLU(true))
  model:add(MaxPooling(2, 2, 2, 2, 0, 0))
  -- 8, 6, 4
  local resolution = 28/2/2
  model:add(nn.View(resolution*resolution*64))
  model:add(nn.Linear(resolution*resolution*64, 320))
  model:add(nn.BatchNormalization(320))
  model:add(cudnn.ReLU(true))
  model:add(nn.Dropout(0.5))
  model:add(nn.Linear(320, 10))
  model:add(cudnn.LogSoftMax())

  if nGPU > 1 then
    assert(nGPU <= cutorch.getDeviceCount(), 'number of GPUs less than nGPU specified')
    local model_single = model
    model = nn.DataParallel(1)
    for i=1,nGPU do
      cutorch.withDevice(i, function() model:add(model_single:clone()) end)
    end
  end

  return model

end
