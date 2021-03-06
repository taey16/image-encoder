
paths.dofile('../dataset.lua')
paths.dofile('../utils/util.lua')
local image_utils = require 'utils.image_utils'

local trainCache = paths.concat(opt.cache, 'trainCache.t7')
local testCache  = paths.concat(opt.cache, 'testCache.t7')
local meanstdCache=paths.concat(opt.cache, 'meanstdCache.t7')

local mean, std

-- Check for existence of opt.data
if not os.execute('cd ' .. opt.data) then
  error(("could not chdir to '%s'"):format(opt.data))
end

-- function to load the image, jitter it appropriately
local trainHook = function(self, path)
  collectgarbage()
  local input = image_utils.loadImage(path, self.loadSize)
  local output = image_utils.random_jitter(input, self.sampleSize)
  output = image_utils.mean_std_norm(output, mean, std)
  return output
end

if paths.filep(trainCache) then
  print('===> Loading train metadata(trainLoader) from cache: '..trainCache)
  trainLoader = torch.load(trainCache)
  trainLoader.sampleHookTrain = trainHook
  assert(trainLoader.paths[1] == paths.concat(opt.data, 'train'),
    'cached files dont have the same path as opt.data. Remove your cached files at: '
    .. trainCache .. ' and rerun the program')
else
  print('===> Creating train metadata(trainLoader)')
  trainLoader = dataLoader{
    paths = {paths.concat(opt.data, 'train')},
    loadSize  = opt.loadSize,
    sampleSize= opt.sampleSize,
    verbose = true,
    forceClasses = opt.forceClasses
  }
  torch.save(trainCache, trainLoader)
  print('===> save done ' .. trainCache)
  trainLoader.sampleHookTrain = trainHook
end
collectgarbage()

-- do some sanity checks on trainLoader
do
  local class = trainLoader.imageClass
  local nClasses = #trainLoader.classes
  assert(class:max() <= nClasses, "class logic has error")
  assert(class:min() >= 1, "class logic has error")
end
-- End of train loader section


-- function to load the image
local testHook = function(self, path)
  collectgarbage()
  local input = image_utils.loadImage(path, self.loadSize)
  local output= image_utils.center_crop(input, self.sampleSize)
  output = image_utils.mean_std_norm(output, mean, std)
  return output
end

if paths.filep(testCache) then
  print('===> Loading test metadata(testLoader) from cache: ' .. testCache)
  testLoader = torch.load(testCache)
  testLoader.sampleHookTest = testHook
  assert(testLoader.paths[1] == paths.concat(opt.data, 'val'),
    'cached files dont have the same path as opt.data. Remove your cached files at: '
    .. testCache .. ' and rerun the program')
else
  print('===> Creating test metadata(testLoader)')
  testLoader = dataLoader{
    paths = {paths.concat(opt.data, 'val')},
    loadSize  = opt.loadSize,
    sampleSize= opt.sampleSize,
    verbose = true,
    -- force consistent class indices between trainLoader and testLoader
    forceClasses = trainLoader.classes
  }
  --print(testLoader)
  torch.save(testCache, testLoader)
  print('===> Save done ' .. testCache)
  testLoader.sampleHookTest = testHook
end
collectgarbage()
-- End of test loader section

-- Estimate the per-channel mean/std 
-- (so that the loaders can normalize appropriately)
if paths.filep(meanstdCache) then
  local meanstd = torch.load(meanstdCache)
  mean = meanstd.mean
  std = meanstd.std
  print('===> Loading mean and std from cache: ' .. meanstdCache)
else
  local tm = torch.Timer()
  local nSamples = 10000
  print('Estimating the mean (per-channel, shared for all pixels) over ' 
        .. nSamples .. ' randomly sampled training images')
  local meanEstimate = {0,0,0}
  for i=1,nSamples do
    local img = trainLoader:sample(1)[1]
    for j=1,3 do
      meanEstimate[j] = meanEstimate[j] + img[j]:mean()
    end
  end
  for j=1,3 do
    meanEstimate[j] = meanEstimate[j] / nSamples
  end
  mean = meanEstimate

  print('Estimating the std (per-channel, shared for all pixels) over ' 
        .. nSamples .. ' randomly sampled training images')
  local stdEstimate = {0,0,0}
  for i=1,nSamples do
    local img = trainLoader:sample(1)[1]
    for j=1,3 do
      stdEstimate[j] = stdEstimate[j] + img[j]:std()
    end
  end
  for j=1,3 do
    stdEstimate[j] = stdEstimate[j] / nSamples
  end
  std = stdEstimate

  local cache = {}
  cache.mean = mean
  cache.std = std
  torch.save(meanstdCache, cache)
  print('===> Save done ' .. meanstdCache)
  print('Time to estimate:', tm:time().real)
end

