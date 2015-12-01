
require 'torch'
require 'loadcaffe'
require 'image'
require 'nn'
paths.dofile('../utils/imagenet_utils.lua')
paths.dofile('../utils/image_utils.lua')

torch.setnumthreads(4)

-- Setting up modelworks and downloading stuff if needed
local proto_name = '/storage/models/vgg/vgg_layer16_deploy.prototxt'
local model_name = '/storage/models/vgg/vgg_layer16.caffemodel'
local backend = 'nn'

print '===> Loading model'
local model = loadcaffe.load(proto_name, model_name, backend)
model.modules[#model.modules] = nil
model:add(nn.SoftMax())
print(model)
model:evaluate()

print '===> Loading synsets'
local synset_words = load_synset()
local image_list, label_list = get_val()

local sampleSize = {3, 224, 224}
local loadSize = {3, 256, 256}

local dataset_root = '/storage/ImageNet/ILSVRC2012/val'
local top1 = 0
local top5 = 0
local trials = 0
for k, fname in ipairs(image_list) do
  print(fname .. ' ' .. label_list[k])
  filename = paths.concat(dataset_root, fname)
  im = image.load(filename)
  label = tonumber(label_list[k]) + 1

  -- Have to resize and convert from RGB to BGR and subtract mean
  input = preprocess(im)
  input = augment_image(input, loadSize, sampleSize)
  scores = model:forward(input):float()
  scores, classes = torch.mean(scores,1):view(-1):sort(true)

  --[[
  -- Propagate through the modelwork and sort outputs in decreasing order and show 5 best classes
  _,classes = model:forward(I):view(-1):float():sort(true)
  --]]

  trials = trials + 1
  top1 = top1 + classes[{{1,1}}]:eq(label):sum()
  top5 = top5 + classes[{{1,5}}]:eq(label):sum()
  io.flush(
    print(("%d top1: %d/%d = %.5f, top5: %d/%d = %.5f"):format(
      k, top1 , trials, top1 / trials * 100, top5, trials, top5 / trials * 100 )
    )
  )
  
end
