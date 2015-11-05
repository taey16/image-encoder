
require 'image'
paths.dofile('../dataset.lua')
paths.dofile('../util.lua')
paths.dofile('../utils/image_utils.lua')

-- This file contains the data-loading logic and details.
-- It is run by each data-loader thread.
------------------------------------------

-- a cache file of the training metadata (if doesnt exist, will be created)
local trainCache = paths.concat(opt.cache, 'trainCache.t7')
local testCache  = paths.concat(opt.cache, 'testCache.t7')
local meanstdCache=paths.concat(opt.cache, 'meanstdCache.t7')

local mean, std

-- Check for existence of opt.data
if not os.execute('cd ' .. opt.data) then
  error(("could not chdir to '%s'"):format(opt.data))
end

-- function to load the image, jitter it appropriately (random crops etc.)
local trainHook = function(self, path)
  collectgarbage()
  local input = loadImage(path)
  local output = random_jitter(input)
  output = mean_std_norm(output)
  return output
end

if paths.filep(trainCache) then
  print('===> Loading train metadata from cache: ' .. trainCache)
  trainLoader = torch.load(trainCache)
  trainLoader.sampleHookTrain = trainHook
  assert(trainLoader.paths[1] == paths.concat(opt.data, 'train'),
         'cached files dont have the same path as opt.data. Remove your cached files at: '
         .. trainCache .. ' and rerun the program')
else
  print('===> Creating train metadata')
  trainLoader = dataLoader{
     paths = {paths.concat(opt.data, 'train')},
     loadSize  = loadSize,
     sampleSize= sampleSize,
     split = 100,
     verbose = true
  }
  print(trainLoader)
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


--------------------------------------------------------------------------------
--[[
   Section 2: Create a test data loader (testLoader),
   which can iterate over the test set and returns an image's
--]]

local function center_crop(input)
  local oH = sampleSize[2]
  local oW = sampleSize[3]
  local iW = input:size(3)
  local iH = input:size(2)
  local w1 = math.ceil((iW-oW)/2)
  local h1 = math.ceil((iH-oH)/2)
  -- center patch
  local output = image.crop(input, w1, h1, w1+oW, h1+oW)
  return output
end

-- function to load the image
local testHook = function(self, path)
  collectgarbage()
  local input = loadImage(path)
  local output= center_crop(input)
  output = mean_std_norm(output)
  return output
end

if paths.filep(testCache) then
  print('===> Loading test metadata from cache: ' .. testCache)
  testLoader = torch.load(testCache)
  testLoader.sampleHookTest = testHook
  assert(testLoader.paths[1] == paths.concat(opt.data, 'val'),
         'cached files dont have the same path as opt.data. Remove your cached files at: '
         .. testCache .. ' and rerun the program')
else
  print('===> Creating test metadata')
  testLoader = dataLoader{
    paths = {paths.concat(opt.data, 'val')},
    loadSize  = loadSize,
    sampleSize= sampleSize,
    split = 0,
    verbose = true,
    -- force consistent class indices between trainLoader and testLoader
    forceClasses = trainLoader.classes
  }
  print(testLoader)
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

