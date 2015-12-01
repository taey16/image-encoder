
require 'loadcaffe'
require 'nn'
paths.dofile('../utils/parallel_utils.lua')

function createModel(nGPU, backend)
  local deploy_file = '/storage/models/vgg/vgg_layer16_deploy.prototxt'
  local weight_file = '/storage/models/vgg/vgg_layer16.caffemodel'
  local backend = backend or 'cudnn'
  local model = loadcaffe.load(deploy_file, weight_file, backend) 
  local LogSoftMax = {}
  if backend == 'nn' then 
    LogSoftMax = nn.LogSoftMax
  else
    require 'cunn'
    require 'cudnn'
    LogSoftMax = cudnn.LogSoftMax
  end

  -- remove nn.SoftMax()
  model:remove(40)
  model:add(LogSoftMax)

  return model
end


--[[
nn.Sequential {
  [input -> (1) -> (2) -> (3) -> (4) -> (5) -> (6) -> (7) -> (8) -> (9) -> (10) -> (11) -> (12) -> (13) -> (14) -> (15) -> (16) -> (17) -> (18) -> (19) -> (20) -> (21) -> (22) -> (23) -> (24) -> (25) -> (26) -> (27) -> (28) -> (29) -> (30) -> (31) -> (32) -> (33) -> (34) -> (35) -> (36) -> (37) -> (38) -> (39) -> (40) -> output]
  (1): nn.SpatialConvolutionMM(3 -> 64, 3x3, 1,1, 1,1)
  (2): nn.ReLU
  (3): nn.SpatialConvolutionMM(64 -> 64, 3x3, 1,1, 1,1)
  (4): nn.ReLU
  (5): nn.SpatialMaxPooling(2,2,2,2)
  (6): nn.SpatialConvolutionMM(64 -> 128, 3x3, 1,1, 1,1)
  (7): nn.ReLU
  (8): nn.SpatialConvolutionMM(128 -> 128, 3x3, 1,1, 1,1)
  (9): nn.ReLU
  (10): nn.SpatialMaxPooling(2,2,2,2)
  (11): nn.SpatialConvolutionMM(128 -> 256, 3x3, 1,1, 1,1)
  (12): nn.ReLU
  (13): nn.SpatialConvolutionMM(256 -> 256, 3x3, 1,1, 1,1)
  (14): nn.ReLU
  (15): nn.SpatialConvolutionMM(256 -> 256, 3x3, 1,1, 1,1)
  (16): nn.ReLU
  (17): nn.SpatialMaxPooling(2,2,2,2)
  (18): nn.SpatialConvolutionMM(256 -> 512, 3x3, 1,1, 1,1)
  (19): nn.ReLU
  (20): nn.SpatialConvolutionMM(512 -> 512, 3x3, 1,1, 1,1)
  (21): nn.ReLU
  (22): nn.SpatialConvolutionMM(512 -> 512, 3x3, 1,1, 1,1)
  (23): nn.ReLU
  (24): nn.SpatialMaxPooling(2,2,2,2)
  (25): nn.SpatialConvolutionMM(512 -> 512, 3x3, 1,1, 1,1)
  (26): nn.ReLU
  (27): nn.SpatialConvolutionMM(512 -> 512, 3x3, 1,1, 1,1)
  (28): nn.ReLU
  (29): nn.SpatialConvolutionMM(512 -> 512, 3x3, 1,1, 1,1)
  (30): nn.ReLU
  (31): nn.SpatialMaxPooling(2,2,2,2)
  (32): nn.View
  (33): nn.Linear(25088 -> 4096)
  (34): nn.ReLU
  (35): nn.Dropout(0.500000)
  (36): nn.Linear(4096 -> 4096)
  (37): nn.ReLU
  (38): nn.Dropout(0.500000)
  (39): nn.Linear(4096 -> 1000)
  (40): nn.SoftMax
}
--]]

