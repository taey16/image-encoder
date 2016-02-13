
require 'torch'
require 'cutorch'
require 'paths'
require 'xlua'
require 'optim'
require 'nn'

--local opts = paths.dofile('opts/opts_mnist.lua')
--local opts = paths.dofile('opts/opts_ilsvrc.lua')
--local opts = paths.dofile('opts/opts_ilsvrc_inception-v3.lua')
local opts = paths.dofile('opts/opts_ilsvrc_inception-v3_random.lua')
--local opts = paths.dofile('opts/opts_clothes.lua')
--local opts = paths.dofile('opts/opts_det.lua')
opt = opts.parse(arg)

--torch.setnumthreads(4)
torch.setdefaulttensortype('torch.FloatTensor')
torch.manualSeed(opt.manualSeed)
cutorch.manualSeed(opt.manualSeed+4322)
--cutorch.setDevice(opt.GPU)

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

