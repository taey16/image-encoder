
require 'torch'
require 'loadcaffe'
require 'image'
require 'nn'
require 'cunn'
require 'cudnn'
paths.dofile('../utils/imagenet_utils.lua')
paths.dofile('../utils/image_utils.lua')

torch.setnumthreads(4)

print '===> Loading model'
local model_filename = '/storage/ImageNet/ILSVRC2012/torch_cache/inception6/image_remove_ThuOct1501:02:042015/model_22.t7'
local model = torch.load(model_filename)
model.modules[#model.modules] = nil
model:add(cudnn.SoftMax())
print(model)
model:cuda()
model:evaluate()

print '===> Loading mean, std' 
local mean_std = torch.load('/storage/ImageNet/ILSVRC2012/torch_cache/meanstdCache.t7')

local dataset_root = '/storage/ImageNet/ILSVRC2012/val'
print '===> Loading synsets'
local synset_words = load_synset()
local image_list, label_list = get_val()


local sampleSize = {3, 224, 224}
local loadSize = {3, 256, 256}

local top1 = 0
local top5 = 0
local trials = 0
for k, fname in ipairs(image_list) do
  --print(fname .. ' ' .. label_list[k])
  filename = paths.concat(dataset_root, fname)
  label = tonumber(label_list[k]) + 1

  -- Have to resize and convert from RGB to BGR and subtract mean
  input = loadImage(filename, loadSize)
  input = mean_std_norm(input, mean_std.mean, mean_std.std)
  --input = augment_image(input, loadSize, sampleSize )
  input = center_crop(input, sampleSize )
  input = input:view(1,input:size(1),input:size(2),input:size(3)):repeatTensor(2,1,1)
  scores = model:forward(input:cuda())
  scores, classes = torch.mean(scores[{1,{}}],1):view(-1):sort(true)
  print(classes[{{1,5}}])
  print(scores[{{1,5}}])

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

