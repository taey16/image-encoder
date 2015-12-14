require 'stn'

function createModel(nGPU)
local spanet=nn.Sequential()
local concat=nn.ConcatTable()
-- first branch is there to transpose inputs to BHWD, for the bilinear sampler
tranet=nn.Sequential()
tranet:add(nn.Identity())
tranet:add(nn.Transpose({2,3},{3,4}))
-- second branch is the localization network
local locnet = nn.Sequential()

-- 224, 336, 336
locnet:add(cudnn.SpatialMaxPooling(2,2,2,2,0,0))
-- 112, 168, 84
locnet:add(cudnn.SpatialConvolution(3,36,3,3,1,1,0,0))
locnet:add(nn.SpatialBatchNormalization(36))
locnet:add(cudnn.ReLU(true))
-- 110, 166, 82
locnet:add(cudnn.SpatialConvolution(36,64,3,3,1,1,0,0))
locnet:add(nn.SpatialBatchNormalization(64))
locnet:add(cudnn.ReLU(true))
-- 108 164, 80
locnet:add(cudnn.SpatialMaxPooling(2,2,2,2,0,0))
-- 54, 82, 40
locnet:add(cudnn.SpatialConvolution(64,128,3,3,1,1,1,1))
locnet:add(nn.SpatialBatchNormalization(128))
locnet:add(cudnn.ReLU(true))
-- 52, 80, 40
locnet:add(cudnn.SpatialConvolution(128,256,3,3,1,1,1,1))
locnet:add(nn.SpatialBatchNormalization(256))
locnet:add(cudnn.ReLU(true))
locnet:add(cudnn.SpatialMaxPooling(2,2,2,2,0,0))
-- 26, 40, 20
locnet:add(cudnn.SpatialConvolution(256,256,3,3,1,1,1,1))
locnet:add(nn.SpatialBatchNormalization(256))
locnet:add(cudnn.ReLU(true))
locnet:add(cudnn.SpatialConvolution(256,256,3,3,1,1,1,1))
locnet:add(nn.SpatialBatchNormalization(256))
locnet:add(cudnn.ReLU(true))
locnet:add(cudnn.SpatialMaxPooling(2,2,2,2,0,0))
-- 13, 20, 10
locnet:add(cudnn.SpatialAveragePooling(13,13,1,1,0,0))
locnet:add(nn.View(256))

-- we initialize the output layer so it gives the identity transform
local outLayer = nn.Linear(256,3)
outLayer.weight:fill(0)
local bias = torch.FloatTensor(3):fill(0)
bias[1]=0.5
bias[2]=0.0
bias[2]=0.0
outLayer.bias:copy(bias)
locnet:add(outLayer)

-- there we generate the grids
locnet:add(nn.AffineTransformMatrixGenerator(false, true, true))
locnet:add(nn.AffineGridGeneratorBHWD(
  opt.sampling_grid_size,
  opt.sampling_grid_size))

-- we need a table input for the bilinear sampler, so we use concattable
concat:add(tranet)
concat:add(locnet)

spanet:add(concat)
spanet:add(nn.BilinearSamplerBHWD())

-- and we transpose back to standard BDHW format for subsequent processing by nn modules
spanet:add(nn.Transpose({3,4},{2,3}))

return spanet

end
