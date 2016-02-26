
require 'torch'
require 'cutorch'
require 'paths'
require 'xlua'
require 'optim'
require 'nn'

--local opts = paths.dofile('opts/opts_mnist.lua')
--local opts = paths.dofile('opts/opts_ilsvrc.lua')
--local opts = paths.dofile('opts/opts_ilsvrc_inception-v3.lua')
--local opts = paths.dofile('opts/opts_ilsvrc_inception-v3_random.lua')
opt = paths.dofile('opts/opts_ilsvrc-inception-v3-random.lua')
--local opts = paths.dofile('opts/opts_clothes.lua')
--local opts = paths.dofile('opts/opts_det.lua')
--opt = opts.parse(arg)

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

