
require 'torch'
require 'cutorch'
require 'image'
require 'nn'
require 'cunn'
require 'cudnn'
paths.dofile('../utils/imagenet_utils.lua')
paths.dofile('../utils/image_utils.lua')

torch.setnumthreads(4)

print '===> Loading model'
local model_filename = 
  '/storage/ImageNet/ILSVRC2012/torch_cache/inception7/digits_gpu_2_lr0.045SatDec514:08:122015/model_40.bn_removed.t7'
  --'/storage/ImageNet/ILSVRC2012/torch_cache/inception7/digits_gpu_2_lr0.045SatDec514:08:122015/model_30.bn_removed.t7'
local original_model = torch.load(model_filename)
local feature_encoder = original_model:get(1)
local classifier = original_model:get(2)
classifier.modules[#classifier.modules] = nil
classifier:add(cudnn.SoftMax())
local model = nn.Sequential()
model:add(feature_encoder):add(classifier)

paths.dofile('../models/vgg16caffe_cudnn.lua')
model_vgg = createModel('cudnn')
model_vgg:remove(#model_vgg)
model_vgg:add(cudnn.SoftMax())

local replica = model
local replica_vgg = model_vgg
model = nn.DataParallelTable(1)
model_vgg = nn.DataParallelTable(1)
for gpu_id=1,2 do
  cutorch.setDevice(gpu_id)
  model:add(replica:clone():cuda(), gpu_id)
  model_vgg:add(replica_vgg:clone():cuda(), gpu_id)
end
cutorch.setDevice(1)
  
print(model)
print(model_vgg)
model:evaluate()
model_vgg:evaluate()
collectgarbage()

print '===> Load classes conf.'
local class_filename = 
  '/storage/ImageNet/ILSVRC2012/torch_cache/inception7/digits_gpu_2_lr0.045SatDec514:08:122015/classes.t7'
class_conf = torch.load(class_filename)
--for i=1, #class_conf do
--  print(class_conf[i])
--end

print '===> Loading mean, std' 
local mean_std = torch.load(
  '/storage/ImageNet/ILSVRC2012/torch_cache/meanstdCache.t7')

local dataset_root = 
  '/storage/ImageNet/ILSVRC2012/val'
print '===> Loading synsets'
local synset_words = load_synset()
local image_list, label_list, synset_list = get_val()

local loadSize = {3, 292, 292}
local sampleSize={3, 256, 256}
local loadSize_vgg = {3, 256, 256}
local sampleSize_vgg={3, 224, 224}

local top1 = 0
local top5 = 0
local trials = 0

classid_match = torch.LongTensor(1000)
for i=1,1000 do
  for k=1,1000 do
    if class_conf[k] == synset_list[i] then
      classid_match[i] = k
      print(i..' '..k)
    end
  end
end

local timer = torch.Timer()

for n, fname in ipairs(image_list) do
  --print(fname .. ' ' .. label_list[n])
  local filename = paths.concat(dataset_root, fname)
  local label = tonumber(label_list[n]) + 1

  local start_loading = timer:time().real
  -- Have to resize and convert from RGB to BGR and subtract mean
  local input = loadImage(filename, loadSize, 0)
  local input1= loadImage(filename, loadSize, 1)
  input = mean_std_norm(input, mean_std.mean, mean_std.std)
  input1= mean_std_norm(input1,mean_std.mean, mean_std.std)
  input = augment_image(input, loadSize, sampleSize )
  input1= augment_image(input1,loadSize, sampleSize )
  local data = torch.FloatTensor(20, sampleSize[1], sampleSize[2], sampleSize[3])
  data[{{1 ,10},{},{},{}}] = input
  data[{{11,20},{},{},{}}] = input1
  local scores, classes
  local elapsed_loading = timer:time().real - start_loading
  local start_process = timer:time().real
  scores = torch.FloatTensor(10, 1000)
  --scores[{{1, 20},{}}] = model:forward(data:cuda()):float()
  --scores, classes = torch.mean(scores,1):view(-1):sort(true)

  input = preprocess(image.load(filename))
  input = augment_image(input, loadSize_vgg, sampleSize_vgg)
  scores_vgg = model_vgg:forward(input:cuda()):float()
  for i=1,1000 do
    scores[{{},{classid_match[i]}}] = scores_vgg[{{},{i}}]
    --scores[{{21,30},{i}}] = scores_vgg[{{},{classid_match[i]}}]
    --scores[{{1,10},{i}}] = scores_vgg[{{},{classid_match[i]}}]
  end
  scores, classes = torch.mean(scores,1):view(-1):sort(true)
  local elapsed_process = timer:time().real - start_process

  trials = trials + 1
  for k=1,5 do
    if k == 1 and class_conf[classes[k]] == synset_list[n] then 
      top1 = top1 + 1
      top5 = top5 + 1
    elseif k > 1  and class_conf[classes[k]] == synset_list[n] then
      top5 = top5 + 1
    end
  end
  io.flush(
    print(("%d top1: %d/%d = %.5f, top5: %d/%d = %.5f %.4f(%.3f)"):format(
      n, top1 , trials, top1 / trials * 100, top5, trials, top5 / trials * 100,
      elapsed_process, elapsed_loading )
    )
  )

end
