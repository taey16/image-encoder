-- wget 'http://torch7.s3-website-us-east-1.amazonaws.com/data/mnist.t7.tgz'
-- tar -xf mnist.t7.tgz

require 'image'
require 'nn'
require 'xlua'

function jittering_RST(img_data, output_resolution)
  local batchSize = img_data:size(1)
  local resolution = img_data:size(3)
  local nChannels = img_data:size(2)
  local rotationFactor = 4
  local scaleFactorFrom = 0.7 
  local scaleFactorTo = 1.2
  local maximumScaledImgSize = 48

  assert(maximumScaledImgSize > resolution * scaleFactorTo)

  local res=torch.FloatTensor(batchSize, nChannels, output_resolution, output_resolution):fill(0)
  local distImg=torch.FloatTensor(1, maximumScaledImgSize, maximumScaledImgSize):fill(0)
  for i = 1, batchSize do
    baseImg = img_data:select(1,i)
    
    R = image.rotate(baseImg, torch.uniform(-3.14/rotationFactor,3.14/rotationFactor))
    scale = torch.uniform(scaleFactorFrom, scaleFactorTo)
    sz = torch.floor(scale*resolution)
    S = image.scale(R, sz, sz)
    rest = maximumScaledImgSize - sz
    offsetx = torch.random(1, 1+rest)
    offsety = torch.random(1, 1+rest)
     
    distImg:zero()
    distImg:narrow(2, offsety, sz):narrow(3, offsetx, sz):copy(S)
    res:select(1,i):copy(image.scale(distImg,output_resolution, output_resolution))
  end
  return res
end

function distortData(foo)
   local res=torch.FloatTensor(foo:size(1), 3, 42, 42):fill(0)
   for i=1,foo:size(1) do
      baseImg=foo:select(1,i)
      distImg=res:select(1,i)
      
      r = image.rotate(baseImg, torch.uniform(-3.14/4,3.14/4))
      scale = torch.uniform(0.7,1.2)
      sz = torch.floor(scale*32)
      s = image.scale(r, sz, sz)
      rest = 42-sz
      offsetx = torch.random(1, 1+rest)
      offsety = torch.random(1, 1+rest)
      
      distImg:narrow(2, offsety, sz):narrow(3,offsetx, sz):copy(s)
   end
   return res
end

function distortData32(foo)
   local res=torch.FloatTensor(foo:size(1), 3, 32, 32):fill(0)
   local distImg=torch.FloatTensor(3, 42, 42):fill(0)
   for i=1,foo:size(1) do
      xlua.progress(i, foo:size(1))
      baseImg=foo:select(1,i)
     
      r = image.rotate(baseImg, torch.uniform(-3.14/4,3.14/4))
      scale = torch.uniform(0.7,1.2)
      sz = torch.floor(scale*32)
      s = image.scale(r, sz, sz)
      rest = 42-sz
      offsetx = torch.random(1, 1+rest)
      offsety = torch.random(1, 1+rest)
      
      distImg:zero()
      distImg:narrow(2, offsety, sz):narrow(3,offsetx, sz):copy(s)
      res:select(1,i):copy(image.scale(distImg,32,32))
   end
   return res
end

function createDatasetsDistorted(dataset_root, batchsize, test_batchsize)

  local numTrainSamples = 50000
  local numTestSamples = 10000
  local train = { 
    data = torch.FloatTensor(numTrainSamples, 3072), 
    labels = torch.FloatTensor(numTrainSamples) 
  }
  for b = 0,4 do
    local train_data_file = paths.concat(dataset_root, 'cifar-10-batches-t7/data_batch_' .. b+1 .. '.t7')
    local subset = torch.load(train_data_file, 'ascii')
    train.data[{{b*10000+1,(b+1)*10000}}] = subset.data:t():float()
    train.labels[{ {b*10000+1, (b+1)*10000} }] = subset.labels:float()
    print('load ' .. train_data_file .. ' done')
  end

  local test_data_file = paths.concat(dataset_root, 'cifar-10-batches-t7/test_batch.t7')
  local subset = torch.load(test_data_file, 'ascii')
  local test = { 
    data  = subset.data:t():float(), 
    labels= subset.labels[1]:float() 
  }
  print('load ' .. test_data_file .. ' done')

  train.data= train.data:reshape(50000, 3, 32, 32)
  test.data = test.data:reshape( 10000, 3, 32, 32)
  train.labels = train.labels + 1
  test.labels = test.labels + 1
  collectgarbage()

  -- distortion   
  if opt.do_distort then
    train.data= distortData32(train.data)
    test.data = distortData32(test.data)
  end

  -- preprocess trainSet
  local normalization = nn.SpatialContrastiveNormalization(1, image.gaussian1D(7))
  for i = 1,train.data:size(1) do
    xlua.progress(i, train.data:size(1))
    -- rgb -> yuv
    local rgb = train.data[i]
    local yuv = image.rgb2yuv(rgb)
    -- normalize y locally:
    yuv[1] = normalization(yuv[{{1}}])
    train.data[i] = yuv
  end
  -- normalize u globally:
  local mean_u= train.data:select(2,2):mean()
  local std_u = train.data:select(2,2):std()
  train.data:select(2,2):add(-mean_u)
  train.data:select(2,2):div(std_u)
  -- normalize v globally:
  local mean_v= train.data:select(2,3):mean()
  local std_v = train.data:select(2,3):std()
  train.data:select(2,3):add(-mean_v)
  train.data:select(2,3):div(std_v)

  -- preprocess testSet
  for i = 1,test.data:size(1) do
    xlua.progress(i, test.data:size(1))
    -- rgb -> yuv
    local rgb = test.data[i]
    local yuv = image.rgb2yuv(rgb)
    -- normalize y locally:
    yuv[1] = normalization(yuv[{{1}}])
    test.data[i] = yuv
  end
  -- normalize u globally:
  test.data:select(2,2):add(-mean_u)
  test.data:select(2,2):div(std_u)
  -- normalize v globally:
  test.data:select(2,3):add(-mean_v)
  test.data:select(2,3):div(std_v)

  --for i = 1,train.data:size(1) do
  --  train.data[i] = image.yuv2rgb(train.data[i])
  --end

  local datasetTrain = {
    getBatch = function(self, idx)
      local data  = train.data:narrow(  1, (idx - 1) * batchsize + 1, batchsize)
      local labels= train.labels:narrow(1, (idx - 1) * batchsize + 1, batchsize)
      return data, labels
    end,
    getNumBatches = function()
      return torch.floor(numTrainSamples / batchsize)
    end
  }
   
  local datasetVal = {
    getBatch = function(self, idx)
      local data  = test.data:narrow(  1, (idx - 1) * test_batchsize + 1, test_batchsize)
      local labels= test.labels:narrow(1, (idx - 1) * test_batchsize + 1, test_batchsize)
      return data, labels
    end,
    getNumBatches = function()
      return torch.floor(numTestSamples / test_batchsize)
    end
  }

  return datasetTrain, datasetVal
end
