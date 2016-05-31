
require 'torch'
require 'paths'

<<<<<<< HEAD
--opt = paths.dofile('opts/opts_ilsvrc-inception-v3-random.lua')
opt = paths.dofile('opts/opts_attribute_button.lua')
=======
opt = paths.dofile('opts/opts_ilsvrc-inception-v3-random.lua')
--opt = paths.dofile('opts/opts_attribute_button.lua')
--opt = paths.dofile('opts/opts_det.lua')
>>>>>>> renewal

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

