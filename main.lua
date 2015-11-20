
require 'torch'
require 'cutorch'
require 'paths'
require 'xlua'
require 'optim'
require 'nn'

local opts = paths.dofile('opts/opts_mnist.lua')
--local opts = paths.dofile('opts/opts_ilsvrc.lua')
opt = opts.parse(arg)

--torch.setnumthreads(4)
torch.setdefaulttensortype('torch.FloatTensor')
torch.manualSeed(opt.manualSeed)
--cutorch.setDevice(opt.GPU)

paths.dofile('data.lua')
paths.dofile('model.lua')
paths.dofile('train.lua')
paths.dofile('test.lua')

-- manually set epoch (useful for retrain)
epoch = opt.epochNumber
if opt.test_initialization then test() end
for i=1,opt.nEpochs do
  train()
  test()
  epoch = epoch + 1
end

