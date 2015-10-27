
require 'cudnn'
require 'cunn'

model = nn.Sequential()

-- building block
local function ConvBNReLU(nInputPlane, nOutputPlane)
  model:add(cudnn.SpatialConvolution(nInputPlane, nOutputPlane, 3,3, 1,1, 1,1))
  model:add(nn.SpatialBatchNormalization(nOutputPlane,1e-3))
  model:add(cudnn.ReLU(true))
  return model
end

-- Will use "ceil" MaxPooling because we want to save as much feature space as we can
local MaxPooling = cudnn.SpatialMaxPooling

-- 32
ConvBNReLU(3,64):add(nn.Dropout(0.3))
ConvBNReLU(64,64)
model:add(MaxPooling(2,2,2,2,0,0))
-- 16
ConvBNReLU(64,128):add(nn.Dropout(0.4))
ConvBNReLU(128,128)
model:add(MaxPooling(2,2,2,2,0,0))
-- 8
ConvBNReLU(128,256):add(nn.Dropout(0.4))
ConvBNReLU(256,256):add(nn.Dropout(0.4))
ConvBNReLU(256,256)
model:add(MaxPooling(2,2,2,2,0,0))
-- 4
ConvBNReLU(256,512):add(nn.Dropout(0.4))
ConvBNReLU(512,512):add(nn.Dropout(0.4))
ConvBNReLU(512,512)
model:add(MaxPooling(2,2,2,2,0,0))
ConvBNReLU(512,512):add(nn.Dropout(0.4))
ConvBNReLU(512,512):add(nn.Dropout(0.4))
ConvBNReLU(512,512)
model:add(MaxPooling(2,2,2,2,0,0))
-- 2
local resolution = 32/2/2/2/2/2
--[[
model:add(cudnn.SpatialConvolution(512, 512, resolution, resolution, 1, 1, 0, 0))
model:add(nn.SpatialBatchNormalization(512,1e-3))
model:add(cudnn.ReLU(true)):add(nn.Dropout(0.5))
model:add(cudnn.SpatialConvolution(512, 10, 1, 1, 1, 1, 0, 0))
model:add(nn.View(10))
model:add(cudnn.LogSoftMax())
--]]
model:add(nn.View(512))
model:add(nn.Dropout(0.5))
model:add(nn.Linear(512,512))
model:add(nn.BatchNormalization(512))
model:add(cudnn.ReLU(true))
model:add(nn.Dropout(0.5))
model:add(nn.Linear(512,10))
model:add(cudnn.LogSoftMax())


MSRinit(model)

