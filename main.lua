
require 'torch'
require 'cutorch'
require 'cudnn'
cudnn.fastest = true
cudnn.benchmark = true
require 'paths'
require 'xlua'
require 'optim'
require 'nn'

--opt = paths.dofile('opts/opts_ilsvrc-inception-v3-random.lua')
opt = paths.dofile('opts/opts_attribute_button.lua')

torch.setdefaulttensortype('torch.FloatTensor')
torch.manualSeed(opt.manualSeed)
cutorch.manualSeedAll(opt.manualSeed)

paths.dofile('model.lua')
paths.dofile('data.lua')
paths.dofile('train.lua')
paths.dofile('test.lua')

epoch = opt.epochNumber
if opt.test_initialization then test() end
for i=1,opt.nEpochs do
  train()
  test()
  epoch = epoch + 1
  collectgarbage()
end

