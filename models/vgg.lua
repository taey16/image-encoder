
require 'cudnn'
require 'cunn'
paths.dofile('init_model_weight.lua')


function createModel()
  -- on a titan black, B/D/E run out of memory even for batch-size 32
  local modelType = 'D'

  -- Create tables describing VGG configurations A, B, D, E
  local cfg = {}
  if modelType == 'A' then
    cfg = {64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'}
  elseif modelType == 'B' then
    cfg = {64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'}
  elseif modelType == 'D' then
    cfg = {64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'}
  elseif modelType == 'E' then
    cfg = {64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'}
  else
    error('Unknown model type: ' .. modelType .. ' | Please specify a modelType A or B or D or E')
  end

  local features = nn.Sequential()
  do
    local iChannels = 3;
    for k,v in ipairs(cfg) do
      if v == 'M' then
        features:add(cudnn.SpatialMaxPooling(2,2,2,2))
      else
        local oChannels = v;
        local conv3 = cudnn.SpatialConvolution(iChannels,oChannels,3,3,1,1,1,1);
        features:add(conv3)
        features:add(nn.SpatialBatchNormalization(oChannels))
        features:add(cudnn.ReLU(true))
        iChannels = oChannels;
      end
    end
  end

  local classifier = nn.Sequential()
  classifier:add(nn.View(512*7*7))
  classifier:add(nn.Linear(512*7*7, 4096))
  classifier:add(nn.BatchNormalization(4096))
  classifier:add(cudnn.ReLU(true))
  classifier:add(nn.Dropout(0.5))
  classifier:add(nn.Linear(4096, 4096))
  classifier:add(nn.BatchNormalization(4096))
  classifier:add(cudnn.ReLU(true))
  classifier:add(nn.Dropout(0.5))
  classifier:add(nn.Linear(4096, opt.nClasses))
  classifier:add(cudnn.LogSoftMax())

  return features, classifier
end

