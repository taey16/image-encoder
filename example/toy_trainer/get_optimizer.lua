
require 'optim'

-- global
-- optim
-- logger_train, logger_test
-- optimState
-- optimizer
os.execute('mkdir -p ' .. opt.save)
os.execute('mkdir -p ' .. opt.save .. '/logs')
os.execute('mkdir -p ' .. opt.save .. '/model')
local logger_train_file= paths.concat(opt.save, 'logs/' .. opt.snapshot_prefix .. 'train.log')
local logger_test_file = paths.concat(opt.save, 'logs/' .. opt.snapshot_prefix .. 'test.log')
logger_train= optim.Logger(logger_train_file)
logger_test = optim.Logger(logger_test_file)
optimState = {}
if opt.optimState then 
  assert(paths.filep(opt.optimState), 'File not found: ' .. opt.optimState)
  print('===> Loading optimState from file: ' .. opt.optimState)
  optimState = torch.load(opt.optimState)
else
  optimState = {
    learningRate = opt.base_lr, 
    learningRateDecay = opt.lr_decay, 
    momentum = opt.momentum, 
    weightDecay = opt.weight_decay
  }
end

