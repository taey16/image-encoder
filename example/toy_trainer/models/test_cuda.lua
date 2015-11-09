
require 'cutorch'
require 'cunn'

local function test_cuda()
  local model = dofile('vgg_bn_drop.lua')
  local batchsize = 32
  local nChannels = 1
  model.base_input_size = 32
  model.nbr_classes = 10
  print(model)
  local criterion = nn.ClassNLLCriterion():cuda()
  -- nn.CrossEntropyCriterion = nn.LogSoftMax _+ nn.ClassNLLCriterion
  -- local criterion = nn.CrossEntropyCriterion():cuda()
  print(criterion)
  local x = torch.Tensor(batchsize, nChannels, model.base_input_size, model.base_input_size):uniform():cuda()
  local y = torch.Tensor(batchsize, model.nbr_classes):bernoulli(0.1):cuda()
  print(string.format('x: %d %d %d %d', x:size(1), x:size(2), x:size(3), x:size(4)))
  print(string.format('y: %d %d', y:size(1), y:size(2)))
  local z = model:forward(x)
  print(string.format('z: %s %s', z:size(1), z:size(2)))
  local df = torch.Tensor(z:size(1), z:size(2)):zero():cuda()
  for i = 1, z:size(1) do
    local loss = criterion:forward(z[i], y[i])
    df[i]:copy(criterion:backward(z[i], y[i])):cuda()
  end
  --[[
  multi-target not supported 
  local loss = criterion:forward(z, y)
  df:copy(criterion:backward(z, y)):cuda()
  --]]
  model:backward(x, df)
  print('CUDA Test Success')
end

torch.setdefaulttensortype('torch.FloatTensor')
print(cutorch.getDeviceProperties(cutorch.getDevice()))
test_cuda()

