require 'stn'

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

local spanet=nn.Sequential()

local concat=nn.ConcatTable()

-- first branch is there to transpose inputs to BHWD, for the bilinear sampler
local tranet=nn.Sequential()
tranet:add(nn.Identity())
tranet:add(nn.Transpose({2,3},{3,4}))

-- second branch is the localization network
local locnet = nn.Sequential()
locnet:add(MaxPooling(2,2,2,2,0,0))
-- 16
locnet:add(SpatialConv(1,20,3,3,1,1,0,0))
locnet:add(nn.SpatialBatchNormalization(20))
-- 14
locnet:add(ReLU(true))
locnet:add(SpatialConv(20,20,3,3,1,1,0,0))
locnet:add(nn.SpatialBatchNormalization(20))
-- 12
locnet:add(ReLU(true))
locnet:add(MaxPooling(2,2,2,2,0,0))
-- 6
locnet:add(SpatialConv(20,20,3,3,1,1,0,0))
locnet:add(nn.SpatialBatchNormalization(20))
-- 4
locnet:add(ReLU(true))
locnet:add(SpatialConv(20,20,3,3,1,1,0,0))
locnet:add(nn.SpatialBatchNormalization(20))
-- 2
locnet:add(ReLU(true))

local resolution = ((32/2 - 2 - 2) / 2) - 2 - 2
locnet:add(nn.View(resolution*resolution*20))
locnet:add(nn.Linear(resolution*resolution*20,20))
locnet:add(nn.BatchNormalization(20))
locnet:add(ReLU(true))

-- we initialize the output layer so it gives the identity transform
local outLayer = nn.Linear(20,6)
outLayer.weight:fill(0)
local bias = torch.FloatTensor(6):fill(0)
bias[1]=1
bias[5]=1
outLayer.bias:copy(bias)
locnet:add(outLayer)

-- there we generate the grids
--locnet:add(nn.View(2,3))
locnet:add(nn.AffineTransformMatrixGenerator())
locnet:add(nn.AffineGridGeneratorBHWD(opt.sampling_grid_size,opt.sampling_grid_size))

-- we need a table input for the bilinear sampler, so we use concattable
concat:add(tranet)
concat:add(locnet)

spanet:add(concat)
spanet:add(nn.BilinearSamplerBHWD())

-- and we transpose back to standard BDHW format for subsequent processing by nn modules
spanet:add(nn.Transpose({3,4},{2,3}))

return spanet

