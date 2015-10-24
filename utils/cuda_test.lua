require 'cutorch'
require 'cunn'
-- require 'very_deep_model.lua'

local function cuda_test()
   local model = paths.dofile('vgg.lua'):cuda()
   -- local model = very_deep_model(24, 10):cuda()
   print(model)
   -- local criterion = nn.MSECriterion():cuda()
   local criterion = nn.ClassNLLCriterion():cuda()
   print(criterion)
   local batchsize = 32
   local x = torch.Tensor(batchsize, 3, model.base_input_size, model.base_input_size):uniform():cuda()
   local y = torch.Tensor(batchsize, model.nbr_classes):bernoulli(0.1):cuda()
   print(string.format('x: %d %d %d %d', x:size(1), x:size(2), x:size(3), x:size(4)))
   print(string.format('y: %d %d', y:size(1), y:size(2)))
   local z = model:forward(x)
   print(string.format('z: %s %s', z:size(1), z:size(2)))
   local df_do = torch.Tensor(z:size(1), y:size(2)):zero()
   for i = 1, z:size(1) do
      local err = criterion:forward(z[i], y[i])
      df_do[i]:copy(criterion:backward(z[i], y[i]))
   end
   model:backward(x, df_do:cuda())
   print("CUDA Test Successful!")
end

torch.setdefaulttensortype('torch.FloatTensor')
print(cutorch.getDeviceProperties(cutorch.getDevice()))
cuda_test()

