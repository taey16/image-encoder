
local M = {}

function M.parse(arg)
  local defaultDir = paths.concat('/storage/mnist/mnist_image')
  local batchsize = 64
  local total_train_samples = 55000
  local network = 'nin'
  local retrain = '/storage/mnist/mnist_image/torch_cache/nin/solverstage_Tue_Oct__6_17:37:40_2015/'

  local cmd = torch.CmdLine()
  cmd:text()
  cmd:text('Options:')
  cmd:option('-cache',defaultDir ..'/torch_cache', 'subdirectory in which to save/log experiments')
  cmd:option('-data', defaultDir .. '/', 'Home of ImageNet dataset')
  cmd:option('-manualSeed',         2, 'Manually set RNG seed')
  cmd:option('-GPU',                1, 'Default preferred GPU')
  cmd:option('-nGPU',               1, 'Number of GPUs to use by default')
  cmd:option('-backend',     'cudnn', 'Options: cudnn | fbcunn | cunn')
  cmd:option('-nDonkeys',        4, 'number of donkeys to initialize (data loading threads)')
  cmd:option('-nEpochs',         100,    'Number of total epochs to run')
  cmd:option('-epochSize',       math.ceil(total_train_samples/batchsize), 'Number of batches per epoch')
  cmd:option('-epochNumber',     60,     'Manual epoch number (useful on restarts)')
  cmd:option('-batchSize',       batchsize,   'mini-batch size (1 = pure stochastic)')
  cmd:option('-test_batchSize',  batchsize,   'test mini-batch size')
  cmd:option('-LR',    0.005, 'learning rate; if set, overrides default LR/WD recipe')
  cmd:option('-momentum',        0.9,  'momentum')
  cmd:option('-weightDecay',     0.00000, 'weight decay')
  cmd:option('-netType',     network, 'Options: alexnet | overfeat')
  cmd:option('-retrain',     '/storage/mnist/mnist_image/torch_cache/nin/solverstage_Tue_Oct__6_17:51:56_2015/model_59.t7', 'provide path to model to retrain with')
  cmd:option('-optimState',  'none', 'provide path to an optimState to reload from')
  cmd:option('-display', 80, '')
  cmd:option('-use_stn', true, '')
  cmd:text()

  local opt = cmd:parse(arg or {})
  -- add commandline specified options
  opt.save = paths.concat(opt.cache, cmd:string(network, opt, {retrain=true, optimState=true, cache=true, data=true}))
  -- add date/time
  opt.save = paths.concat(opt.save, 'solverstate_' .. os.date():gsub(' ','_'))

  print('===> Saving everything to: ' .. opt.save)
  os.execute('mkdir -p ' .. opt.save)

  return opt
end

return M

