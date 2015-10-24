
require 'cudnn'
require 'cunn'

local MaxPooling = cudnn.SpatialMaxPooling

local model = nn.Sequential()
-- 32, 24, 16
model:add(cudnn.SpatialConvolution(3, 16, 3, 3, 1, 1, 1, 1))
model:add(cudnn.ReLU(true))
--model:add(nn.Dropout(0.3))
model:add(cudnn.SpatialConvolution(16, 16, 3, 3, 1, 1, 1, 1))
model:add(cudnn.ReLU(true))
--model:add(nn.Dropout(0.3))
model:add(cudnn.SpatialConvolution(16, 32, 1, 1, 1, 1, 0, 0))
model:add(cudnn.ReLU(true))
model:add(cudnn.SpatialConvolution(32, 48, 1, 1, 1, 1, 0, 0))
model:add(cudnn.ReLU(true))
model:add(MaxPooling(2, 2, 2, 2, 0, 0))
-- 16, 12, 8
model:add(cudnn.SpatialConvolution(48, 48, 3, 3, 1, 1, 1, 1))
model:add(cudnn.ReLU(true))
--model:add(nn.Dropout(0.4))
model:add(cudnn.SpatialConvolution(48, 48, 3, 3, 1, 1, 1, 1))
model:add(cudnn.ReLU(true))
--model:add(nn.Dropout(0.4))
model:add(cudnn.SpatialConvolution(48, 64, 1, 1, 1, 1, 0, 0))
model:add(cudnn.ReLU(true))
model:add(cudnn.SpatialConvolution(64, 64, 1, 1, 1, 1, 0, 0))
model:add(cudnn.ReLU(true))
model:add(MaxPooling(2, 2, 2, 2, 0, 0))
-- 8, 6, 4
local resolution = opt.sampling_grid_size/2/2
--[[
model:add(cudnn.SpatialConvolution(64, 320, resolution, resolution, 1, 1, 0, 0))
model:add(cudnn.ReLU(true)):add(nn.Dropout(0.5))
model:add(cudnn.SpatialConvolution(320, 10, 1, 1, 1, 1, 0, 0))
model:add(nn.View(10))
model:add(cudnn.LogSoftMax())
--]]

model:add(nn.View(resolution*resolution*64))
model:add(nn.Linear(resolution*resolution*64, 320))
model:add(nn.BatchNormalization(320))
model:add(cudnn.ReLU(true))
model:add(nn.Dropout(0.5))
model:add(nn.Linear(320, 10))
model:add(cudnn.LogSoftMax())

return model
