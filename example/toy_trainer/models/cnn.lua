
require 'cudnn'
require 'cunn'

local MaxPooling = cudnn.SpatialMaxPooling

model = nn.Sequential()
-- 32, 24, 16
model:add(cudnn.SpatialConvolution(1, 32, 9, 9, 1, 1, 0, 0))
model:add(cudnn.ReLU(true))
model:add(MaxPooling(2, 2, 2, 2, 0, 0))
model:add(cudnn.SpatialConvolution(32, 64, 7, 7, 1, 1, 0, 0))
model:add(cudnn.ReLU(true))
model:add(MaxPooling(2, 2, 2, 2, 0, 0))

local resolution = (((40 - 8) / 2) - 6) / 2

model:add(cudnn.SpatialConvolution(64, 10, resolution, resolution, 1, 1, 0, 0))
model:add(nn.Dropout(0.5))
model:add(nn.View(10))
model:add(cudnn.LogSoftMax())

--[[
model:add(nn.View(resolution*resolution*64))
model:add(nn.Linear(resolution*resolution*64, 320))
model:add(nn.BatchNormalization(320,1e-5,0.9))
model:add(cudnn.ReLU(true))
model:add(nn.Dropout(0.5))
model:add(nn.Linear(320, 10))
model:add(cudnn.LogSoftMax())
--]]

-- initialization from MSR
local function MSRinit(net)
  local function init(name)
    for k, v in pairs(net:findModules(name)) do
      local n = v.kW*v.kH*v.nOutputPlane
      print('k, v: ', k, n)
      v.weight:normal(0,math.sqrt(2/n))
      v.bias:zero()
    end
  end
  init'nn.SpatialConvolution'
  init'cudnn.SpatialConvolution'
end
MSRinit(model)

