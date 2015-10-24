
require 'image'
paths.dofile('../dataset.lua')
paths.dofile('../util.lua')

-- This file contains the data-loading logic and details.
-- It is run by each data-loader thread.
------------------------------------------

-- a cache file of the training metadata (if doesnt exist, will be created)
local trainCache = paths.concat(opt.cache, 'trainCache.t7')
local testCache  = paths.concat(opt.cache, 'testCache.t7')
local meanstdCache=paths.concat(opt.cache, 'meanstdCache.t7')

local loadSize   = {3, 256, 256}
local sampleSize = {3, 224, 224}
-- channel-wise mean and std.
local mean, std

-- Check for existence of opt.data
if not os.execute('cd ' .. opt.data) then
  error(("could not chdir to '%s'"):format(opt.data))
end

local function resize_crop(input)
  if torch.uniform() > 0.5 then
    -- find the smaller dimension, and 
    -- resize it to 256 (while keeping aspect ratio)
    local iW = input:size(3)
    local iH = input:size(2)
    local output = torch.FloatTensor()
    if iW < iH then
      output = image.scale(input, loadSize[2], loadSize[2] * iH / iW)
    else
      output = image.scale(input, loadSize[3] * iW / iH, loadSize[3])
    end
  else
    -- resize it to 256 (while breaking aspect ratio)
    output = image.scale(input, loadSize[2], loadSize[3])
  end
  return output
end

local function loadImage(path)
  local input = image.load(path)
  -- 1-channel image loaded as 2D tensor
  if input:dim() == 2 then
    input = input:view(1,input:size(1), input:size(2)):repeatTensor(3,1,1)
  -- 1-channel image
  elseif input:dim() == 3 and input:size(1) == 1 then
    input = input:repeatTensor(3,1,1)
  elseif input:dim() == 3 and input:size(1) == 3 then 
    -- 3-channel image
  elseif input:dim() == 3 and input:size(1) == 4 then 
    -- image with alpha
    input = input[{{1,3},{},{}}]
  else
    print(#input)
    error('not 2-channel or 3-channel image')
  end

  if torch.uniform() > 0.5 then
    -- find the smaller dimension, and 
    -- resize it to 256 (while keeping aspect ratio)
    local iW = input:size(3)
    local iH = input:size(2)
    if iW < iH then
      input = image.scale(input, loadSize[2], loadSize[2] * iH / iW)
    else
      input = image.scale(input, loadSize[3] * iW / iH, loadSize[3])
    end
  else
    -- resize it to 256 (while breaking aspect ratio)
    input = image.scale(input, loadSize[2], loadSize[3])
  end
  return input
end


--------------------------------------------------------------------------------
--[[
   Section 1: Create a train data loader (trainLoader),
   which does class-balanced sampling from the dataset and does a random crop
--]]

local function random_flip(input)
  if torch.uniform() > 0.5 then
    input = image.hflip(input)
  end
  return input
end

local function random_jitter(input)
  local iW = input:size(3)
  local iH = input:size(2)
  local oW = sampleSize[3]
  local oH = sampleSize[2]
  local h1 = math.ceil(torch.uniform(1e-2, iH-oH))
  local w1 = math.ceil(torch.uniform(1e-2, iW-oW))
  local output = image.crop(input, w1, h1, w1 + oW, h1 + oH)
  assert(output:size(3) == oW)
  assert(output:size(2) == oH)

  output = random_flip(output)

  return output
end

local function mean_std_norm(input)
  -- mean/std
  for i=1,3 do -- channels
    if mean then input[{{i},{},{}}]:add(-mean[i]) end
    if  std then input[{{i},{},{}}]:div(std[i]) end
  end
  return input 
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

