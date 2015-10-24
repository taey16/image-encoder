
require 'torch'
require 'cutorch'
require 'paths'
require 'xlua'
require 'optim'
require 'nn'
paths.dofile('fbcunn_files/AbstractParallel.lua')
paths.dofile('fbcunn_files/ModelParallel.lua')
paths.dofile('fbcunn_files/DataParallel.lua')
paths.dofile('fbcunn_files/Optim.lua')

-- local opts = paths.dofile('opts/opts_ilsvrc.lua')
local opts = paths.dofile('opts/opts_det.lua')
opt = opts.parse(arg)

torch.setnumthreads(4)
torch.setdefaulttensortype('torch.FloatTensor')
torch.manualSeed(opt.manualSeed)
cutorch.setDevice(opt.GPU)

print('===> Saving everything to: ' .. opt.save)
os.execute('mkdir -p ' .. opt.save)

paths.dofile('data.lua')
paths.dofile('model.lua')
paths.dofile('train.lua')
paths.dofile('test.lua')
paths.dofile('util.lua')

-- manually set epoch (useful for retrain)
epoch = opt.epochNumber
if opt.test_initialization then test() end
for i=1,opt.nEpochs do
   train()
   test()
   epoch = epoch + 1
end
