
local M = {}

function M.parse(arg)
  local defaultDir= paths.concat('/storage/ImageNet/ILSVRC2012/')
  local cache_dir = paths.concat(defaultDir, 'torch_cache');
  local data_dir  = paths.concat(defaultDir, './')
  local batchsize = 20
  local total_train_samples = 1281167 - 1
  local network = 'inception6' --'vgg16caffe'
  
  local backend = 'cudnn'
  local donkey_filename = 'donkey_ilsvrc12.lua'
  local retrain_path = '/storage/ImageNet/ILSVRC2012/torch_cache/inception6/stn_TueOct2023:01:282015/'
  if retrain_path then
    initial_model = paths.concat(retrain_path, 'model_28.t7') 
    initial_optimState = paths.concat(retrain_path, 'optimState_28.t7')
  else
    initial_model = nil
    initial_optimState = nil
  end

  local cmd = torch.CmdLine()
  cmd:text()
  cmd:text('Options:')

  cmd:option('-cache', cache_dir, 'subdirectory in which to save/log experiments')
  cmd:option('-data', data_dir, 'root of dataset')
  cmd:option('-nDonkeys', 4, 'number of donkeys to initialize (data loading threads)')
  cmd:option('-donkey_filename', ('donkey/%s'):format(donkey_filename), '')

  cmd:option('-manualSeed', 222, 'Manually set RNG seed')

  cmd:option('-GPU', 1, 'Default preferred GPU')
  cmd:option('-nGPU', 1, 'Number of GPUs to use by default')
  cmd:option('-backend', backend, 'Options: cudnn | fbcunn | cunn')


  cmd:option('-nEpochs', 100, 'Number of total epochs to run')
  cmd:option('-epochSize', math.ceil(total_train_samples/batchsize), 'Number of batches per epoch')
  cmd:option('-epochNumber', 29,'Manual epoch number (useful on restarts)')
  cmd:option('-batchSize', batchsize, 'mini-batch size (1 = pure stochastic)')
  cmd:option('-test_batchSize', batchsize, 'test mini-batch size')
  cmd:option('-test_initialization', false, 'test_initalization')

  cmd:option('-LR', 0.0005, 'learning rate; if set, overrides default LR/WD recipe')
  cmd:option('-momentum', 0.9,  'momentum')
  cmd:option('-weightDecay', 0.00000, 'weight decay')

  cmd:option('-use_stn', true, '')
  cmd:option('-netType', network, 'Options: alexnet | overfeat')
  cmd:option('-retrain', initial_model, 'provide path to model to retrain with')

  cmd:option('-optimState', initial_optimState, 'provide path to an optimState to reload from')

  cmd:option('-display', 10, 'interval for printing train loss per minibatch')
  cmd:option('-snapshot', 4000, 'interval for conditional_save')
  cmd:text()

  local opt = cmd:parse(arg or {})
  -- add commandline specified options
  opt.save = paths.concat(opt.cache, cmd:string(network, opt, {retrain=true, optimState=true, cache=true, data=true}))
  opt.save = paths.concat(opt.save, 'stn_' .. os.date():gsub(' ',''))
  return opt
end

-- return nil initially
return M

