-- wget 'http://torch7.s3-website-us-east-1.amazonaws.com/data/mnist.t7.tgz'
-- tar -xf mnist.t7.tgz

require 'image'

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
   local res=torch.FloatTensor(foo:size(1), 1, 42, 42):fill(0)
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

function distortData40(foo)
   local res=torch.FloatTensor(foo:size(1), 1, 40, 40):fill(0)
   for i=1,foo:size(1) do
      baseImg=foo:select(1,i)
      distImg=res:select(1,i)
      
      r = image.rotate(baseImg, torch.uniform(-3.14/4,3.14/4))
      scale = torch.uniform(0.7,1.2)
      sz = torch.floor(scale*32)
      s = image.scale(r, sz, sz)
      rest = 40-sz
      offsetx = torch.random(1, 1+rest)
      offsety = torch.random(1, 1+rest)
      
      distImg:narrow(2, offsety, sz):narrow(3,offsetx, sz):copy(s)
   end
   return res
end

function distortData32(foo)
   local res=torch.FloatTensor(foo:size(1), 1, 32, 32):fill(0)
   local distImg=torch.FloatTensor(1, 42, 42):fill(0)
   for i=1,foo:size(1) do
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

function createDatasetsDistorted(dataset_root, batchsize, testbatchsize)
   local testFileName = paths.concat(dataset_root, 'mnist.t7/test_32x32.t7')
   local trainFileName= paths.concat(dataset_root, 'mnist.t7/train_32x32.t7')
   local train= torch.load(trainFileName, 'ascii')
   local test = torch.load(testFileName, 'ascii')
   print('===> Load mnist done')
   train.data  = train.data:float()
   test.data   = test.data:float()
   if opt.use_cuda then
     train.labels= train.labels:float()
     test.labels = test.labels:float()
   else
     train.labels= train.labels:long()
     test.labels = test.labels:long()
   end
   
   -- distortion   
   if opt.do_distort then
     train.data= distortData32(train.data)
     test.data = distortData32(test.data)
   end

   -- get mean std
   local mean = train.data:mean()
   local std = train.data:std()
   train.data:add(-mean):div(std)
   test.data:add(-mean):div(std)
   
   local batchSize = batchsize
   local test_batchSize = testbatchsize
   local numTrainSamples = 60000
   local numTestSamples = 10000
   
   local datasetTrain = {
      getBatch = function(self, idx)
         local data = train.data:narrow(1, (idx - 1) * batchSize + 1, batchSize)
         local labels = train.labels:narrow(1, (idx - 1) * batchSize + 1, batchSize)
         return data, labels, batchSize
      end,
      getNumBatches = function()
         return torch.floor(numTrainSamples / batchSize)
      end
   }
   
   local datasetVal = {
      getBatch = function(self, idx)
         local data = test.data:narrow(1, (idx - 1) * test_batchSize + 1, test_batchSize)
         local labels = test.labels:narrow(1, (idx - 1) * test_batchSize + 1, test_batchSize)
         return data, labels, test_batchSize
      end,
      getNumBatches = function()
         return torch.floor(numTestSamples / test_batchSize)
      end
   }
   
   return datasetTrain, datasetVal
end
