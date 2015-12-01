
require 'torch'
paths.dofile('../utils/imagenet_utils.lua')
paths.dofile('../utils/image_utils.lua')
paths.dofile('../models/vgg16caffe_cudnn.lua')

torch.setnumthreads(4)
local use_cuda = true
local backend
if use_cuda then
  require 'cutorch'
  require 'cunn'
  require 'cudnn'
  backend = 'cudnn'
else
  backend = 'nn'
end

local model = createModel(1, backend)
model:remove(#model)
model:add(cudnn.SoftMax())
local replica = model
model = nn.DataParallelTable(1)
for gpu_id=3,4 do
  cutorch.setDevice(gpu_id)
  model:add(replica:clone():cuda(), gpu_id)
end
cutorch.setDevice(3)
print(model)
model:evaluate()

print '===> Loading synsets'
local synset_words = load_synset()
local image_list, label_list, synset_list = get_val()

local dataset_root = '/storage/ImageNet/ILSVRC2012/val'
local loadSize  ={3, 256, 256}
local sampleSize={3, 224, 224}
local top1 = 0
local top5 = 0
local trials = 0
for k, fname in ipairs(image_list) do
  --print(fname .. ' ' .. label_list[k])
  local filename = paths.concat(dataset_root, fname)
  local timer = torch.Timer()
  local im = image.load(filename)
  local label = tonumber(label_list[k]) + 1

  -- Have to resize and convert from RGB to BGR and subtract mean
  local input = preprocess(im)
  input = augment_image(input, loadSize, sampleSize)
  local scores= model:forward(input):float()
  scores, classes = torch.mean(scores,1):view(-1):sort(true)
  local elapsed = timer:time().real

  --[[
  -- Propagate through the modelwork and sort outputs in decreasing order and show 5 best classes
  _,classes = model:forward(I):view(-1):float():sort(true)
  --]]

  trials = trials + 1
  top1 = top1 + classes[{{1,1}}]:eq(label):sum()
  top5 = top5 + classes[{{1,5}}]:eq(label):sum()
  io.flush(
    print(("%d top1: %d/%d = %.5f, top5: %d/%d = %.5f in %.4f"):format(
      k, top1 , trials, top1 / trials * 100, top5, trials, top5 / trials * 100, 
      elapsed )
    )
  )
  
end
