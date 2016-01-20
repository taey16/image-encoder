
require 'torch'
require 'cutorch'
require 'image'
require 'nn'
require 'cunn'
require 'cudnn'
paths.dofile('../../utils/imagenet_utils.lua')
paths.dofile('../../utils/image_utils.lua')
paths.dofile('/works/nt2/misc/optim_updates.lua')

print '===> Loading model'
local model_filename = 
  '/storage/ImageNet/ILSVRC2012/torch_cache/inception6/gpu_2_lr0.045ThuNov2612:23:162015/model_29.t7'
  --'/storage/product/det/torch_cache/inception6/det_stnThuDec318:29:322015/model_29.bn_removed.t7'
local model = torch.load(model_filename)
local feature = model:get(1):get(1)
--local feature = model:get(1)
local classifier = model:get(2)
model = nn.Sequential():add(feature):add(classifier)
model:get(2).modules[#model:get(2)] = nil
model:get(2):add(cudnn.LogSoftMax())
print(model)
model:cuda()
criterion = nn.ClassNLLCriterion():cuda()

print '===> Loading mean, std' 
local mean_std_filename = 
  '/storage/product/det/torch_cache/meanstdCache.t7'
local mean_std = torch.load(mean_std_filename)

local loadSize  = {3, 256, 256}
local sampleSize= {3, 224, 224}

local inputs = torch.FloatTensor(1, sampleSize[1], sampleSize[2], sampleSize[3])
--inputs[{1,{1},{},{}}]:normal(mean_std.mean[1], mean_std.std[1])
--inputs[{1,{2},{},{}}]:normal(mean_std.mean[2], mean_std.std[2])
--inputs[{1,{3},{},{}}]:normal(mean_std.mean[3], mean_std.std[3])
inputs:normal(-0.01, 0.01)
local label = torch.FloatTensor(1):fill(0)
label[1] = 5
print(label)


function normalize_filter(filter)
  for i=1,64 do
    local filt = filter[{i,{},{},{}}]:clone()
    min_r = filt[{1,{},{}}]:min()
    min_g = filt[{2,{},{}}]:min()
    min_b = filt[{3,{},{}}]:min()
    max_r = filt[{1,{},{}}]:max()
    max_g = filt[{2,{},{}}]:max()
    max_b = filt[{3,{},{}}]:max()

    filt[{1,{},{}}] = filt[{1,{},{}}]:add(-min_r):div(max_r - min_r)
    filt[{2,{},{}}] = filt[{2,{},{}}]:add(-min_g):div(max_g - min_g)
    filt[{3,{},{}}] = filt[{3,{},{}}]:add(-min_b):div(max_b - min_b)
    filter[{i,{},{},{}}] = filt
  end

  return filter
end

--input_img = loadImage('/works/caffe_build_sys_py/examples/images/cat.jpg', sampleSize)
--inputs[{1,{},{},{}}] = input_img

inputs = inputs:cuda()
label = label:cuda()

local loss, outputs, inputs_grad
local optim_state = {}
local lr = 0.0001
local wc = 0.0001
local beta1 = 0.9
local beta2 = 0.999
local mom = 0.9
local epsilon = 1e-8
local grad_clip = 0.01
local iter = 0
local max_iter = 1000000000
while iter < max_iter do
  model:zeroGradParameters()
  inputs = inputs:resize(1,3,28,28)
  inputs = inputs:resize(1,3,224,224)
  outputs = model:forward(inputs)
  loss = criterion:forward(outputs, label)
  local gradOutputs = criterion:backward(outputs, label)
  model:backward(inputs, gradOutputs)
  inputs_grad = model:get(1):get(1).gradInput
  --inputs_grad:clamp(-grad_clip,grad_clip)
  inputs_grad:add(wc, inputs)
  --adam(inputs, inputs_grad, lr, beta1, beta2, epsilon, optim_state)
  sgdmom(inputs, inputs_grad, lr, mom, optim_state)

  outputs_str = ''
  for i = 1,16 do
    outputs_str = string.format('%s %f', outputs_str, math.exp(outputs[i]))
  end
  io.flush(print(string.format(
    'iter: %d, loss: %f, norm: %f [%s]', iter, loss, inputs_grad:norm(), outputs_str
  )))
  if iter % 200 == 0 then
    local inputs_copy = inputs:float():clone()
    --[[
    inputs_copy[{1,{1},{},{}}]:div(1.0/mean_std.std[1])
    inputs_copy[{1,{2},{},{}}]:div(1.0/mean_std.std[2])
    inputs_copy[{1,{3},{},{}}]:div(1.0/mean_std.std[3])
    inputs_copy[{1,{1},{},{}}]:add(mean_std.mean[1])
    inputs_copy[{1,{2},{},{}}]:add(mean_std.mean[2])
    inputs_copy[{1,{3},{},{}}]:add(mean_std.mean[3])
    --]]
    
    local output_image= inputs_copy[{1,{},{},{}}]:clone()
    local r_min_value = inputs_copy[{1,{1},{},{}}]:min()
    local g_min_value = inputs_copy[{1,{2},{},{}}]:min()
    local b_min_value = inputs_copy[{1,{3},{},{}}]:min()
    local r_max_value = inputs_copy[{1,{1},{},{}}]:max()
    local g_max_value = inputs_copy[{1,{2},{},{}}]:max()
    local b_max_value = inputs_copy[{1,{3},{},{}}]:max()
    local output_image= inputs_copy[{1,{},{},{}}]:clone()
    output_image[{1,{},{}}]:add(-r_min_value):div(r_max_value - r_min_value)
    output_image[{2,{},{}}]:add(-g_min_value):div(g_max_value - g_min_value)
    output_image[{3,{},{}}]:add(-b_min_value):div(b_max_value - b_min_value)
    local filename = string.format(
      'label_%02d_neuron_iter_%06d.png', label[1] , iter
    )
    save_images(output_image, 3, filename)
    --[[
    local filter = model:get(1):get(1).weight:clone():float()
    print(filter:size())
    filename = string.format(
      'filter_layer0.png'
    )
    filter = normalize_filter(filter)
    save_images(filter, 64, filename)
    --]]
  end
  iter = iter + 1
end

