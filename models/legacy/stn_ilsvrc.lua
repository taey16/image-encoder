require 'stn'
--paths.dofile('init_model_weight.lua')

function createModel()
  local spanet=nn.Sequential()
  local concat=nn.ConcatTable()
  -- first branch is there to transpose inputs to BHWD, 
  -- for the bilinear sampler
  tranet=nn.Sequential()
  tranet:add(nn.Identity())
  tranet:add(nn.Transpose({2,3},{3,4}))
  -- second branch is the localization network
  local locnet = nn.Sequential()

  -- 224, 448
  locnet:add(cudnn.SpatialMaxPooling(2,2,2,2,0,0))
  -- 112, 224
  locnet:add(cudnn.SpatialConvolution(3,36,3,3,1,1,0,0))
  locnet:add(nn.SpatialBatchNormalization(36))
  locnet:add(cudnn.ReLU(true))
  -- 110, 222
  locnet:add(cudnn.SpatialConvolution(36,64,3,3,1,1,0,0))
  locnet:add(nn.SpatialBatchNormalization(64))
  locnet:add(cudnn.ReLU(true))
  -- 108, 220
  locnet:add(cudnn.SpatialMaxPooling(2,2,2,2,0,0))
  -- 54,  110
  locnet:add(cudnn.SpatialConvolution(64,128,3,3,1,1,0,0))
  locnet:add(nn.SpatialBatchNormalization(128))
  locnet:add(cudnn.ReLU(true))
  -- 52, 108
  locnet:add(cudnn.SpatialConvolution(128,256,3,3,1,1,1,1))
  locnet:add(nn.SpatialBatchNormalization(256))
  locnet:add(cudnn.ReLU(true))
  locnet:add(cudnn.SpatialMaxPooling(2,2,2,2,0,0))
  -- 26, 54
  locnet:add(cudnn.SpatialConvolution(256,256,3,3,1,1,1,1))
  locnet:add(nn.SpatialBatchNormalization(256))
  locnet:add(cudnn.ReLU(true))
  locnet:add(cudnn.SpatialConvolution(256,256,3,3,1,1,1,1))
  locnet:add(nn.SpatialBatchNormalization(256))
  locnet:add(cudnn.ReLU(true))
  locnet:add(cudnn.SpatialMaxPooling(2,2,2,2,0,0))
  -- 13, 27

  locnet:add(cudnn.SpatialMaxPooling(27,27,1,1,0,0))
  locnet:add(nn.View(256))
  locnet:add(nn.Linear(256,16))
  locnet:add(nn.BatchNormalization(16))
  locnet:add(cudnn.ReLU(true))

  -- we initialize the output layer so it gives the identity transform
  local regression_layer = nn.Linear(16,6)
  regression_layer.weight:fill(0)
  local bias = torch.FloatTensor(6):fill(0)
  bias[1]=0.9
  bias[5]=0.9
  --bias[4]=0
  regression_layer.bias:copy(bias)
  locnet:add(regression_layer)

  -- there we generate the grids
  locnet:add(nn.AffineTransformMatrixGenerator())
  locnet:add(nn.AffineGridGeneratorBHWD(
    opt.sampling_grid_size,
    opt.sampling_grid_size))

  -- we need a table input for the bilinear sampler, 
  -- so we use concattable
  concat:add(tranet)
  concat:add(locnet)

  spanet:add(concat)
  spanet:add(nn.BilinearSamplerBHWD())

  -- and we transpose back to standard BDHW format for 
  -- subsequent processing by nn modules
  spanet:add(nn.Transpose({3,4},{2,3}))

  --MSRinit( spanet )

  return spanet

end

