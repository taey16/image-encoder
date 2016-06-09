
require 'torch'
require 'paths'

--opt = paths.dofile('opts/opts_ilsvrc-inception-v3-random.lua')
opt = paths.dofile('opts/opts_ilsvrc-resception-default.lua') -- default opts for resception

paths.dofile('model.lua') -- creating model
paths.dofile('data.lua') -- perpare ilsvrc2012 train/test cache
paths.dofile('train.lua') -- include functions globally for training
paths.dofile('test.lua') -- include functions globally for testing

epoch = opt.epochNumber
if opt.test_initialization then test() end -- testing first if you need
for i=1,opt.nEpochs do
  -- main loop
  train()
  test()
  epoch = epoch + 1
  collectgarbage()
end

