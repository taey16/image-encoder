
require 'torch'
require 'cutorch'
require 'image'
require 'nn'
require 'cunn'
require 'cudnn'
require 'stn'
paths.dofile('../utils/imagenet_utils.lua')
paths.dofile('../utils/image_utils.lua')


torch.setnumthreads(4)
cutorch.setDevice(2)

print '===> Loading model'
local model_filename = '/home/taey16/storage/product/det/torch_cache/inception6/stn_epoch28/model_69.t7'
local model = torch.load(model_filename)
model.modules[#model.modules] = nil
model:add(cudnn.SoftMax())
print(model)
model:cuda()
model:evaluate()
--]]

print '===> Loading mean, std' 
local mean_std = torch.load('/home/taey16/storage/ImageNet/ILSVRC2012/torch_cache/meanstdCache.t7')

--[[
print '===> Loading synsets'
local dataset_root = '/storage/ImageNet/ILSVRC2012/val'
local synset_words = load_synset()
local image_list, label_list = get_val()
--]]

local loadSize  = {3, 256, 256}
local sampleSize= {3, 224, 224}

local top1 = 0
local top5 = 0
local trials = 0
--for k, fname in ipairs(image_list) do
  --print(fname .. ' ' .. label_list[k])
  --fname = '/home/taey16/works/image-encoder/img/1171563108_B_V.jpg'
  -- fname = '/home/taey16/works/image-encoder/img/Casual-Shirts-Brands-in-India.jpg'
  fname = '/home/taey16/works/image-encoder/img/54839Formal Shirts 03.jpg'
  filename = fname
  --filename = paths.concat(dataset_root, fname)
  --label = tonumber(label_list[k]) + 1

  -- Have to resize and convert from RGB to BGR and subtract mean
  -- require('mobdebug').start()
  input = loadImage(filename, loadSize)
  input = mean_std_norm(input, mean_std.mean, mean_std.std)
  input = augment_image(input, loadSize, sampleSize )
  --input = center_crop(input, sampleSize )
  print(input:size())
  local scores = model:forward(input:cuda()):float()
  scores, classes = torch.mean(scores,1):view(-1):sort(true)
  save_images(model.modules[1].output, 10)
  trials = trials + 1

--end

