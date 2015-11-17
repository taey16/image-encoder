
local M = {}

function M.parse(arg)
  local defaultDir= '/storage/mnist/mnist_image'
  local cache_dir = paths.concat(defaultDir, 'torch_cache');
  local data_dir  = paths.concat(defaultDir, './')
  local batchsize = 64
  local test_batchsize = 50
  local total_train_samples = 55000
  local network = 'inception6' --'vgg' --'inception6' --'vgg16caffe'
  local sampleSize= {3, 224, 224}
  local loadSize  = {3, 256, 256}
  local nGPU = {1, 2}
  local current_epoch = 1
  local test_initialization = true
  local exp_name = 'gpu_1'
  
  local backend = 'cudnn'
  local retrain_path = nil
  if retrain_path then
    initial_model = paths.concat(retrain_path, 'model_?.t7') 
    initial_optimState = paths.concat(retrain_path, 'optimState_?.t7')
  else
    initial_model = false
    initial_optimState = false
  end

  local cmd = torch.CmdLine()
  cmd:text()
  cmd:text('Options:')

  cmd:option('-cache', cache_dir, 'subdirectory in which to save/log experiments')
  cmd:option('-data', data_dir, 'root of dataset')
  cmd:option('-nDonkeys', 0, 'number of donkeys to initialize (data loading threads)')
  cmd:option('-donkey_filename', 'donkey/donkey.lua', 'donkey file')

  cmd:option('-manualSeed', 1123, 'Manually set RNG seed')

  cmd:option('-GPU', 1, 'Default preferred GPU')
  --cmd:option('-nGPU', 2, 'Number of GPUs to use by default')
  cmd:option('-backend', backend, 'Options: cudnn | fbcunn | cunn')

  cmd:option('-nEpochs', 1000, 'Number of total epochs to run')
  cmd:option('-epochSize', math.ceil(total_train_samples/batchsize), 'Number of batches per epoch')
  cmd:option('-epochNumber', current_epoch,'Manual epoch number (useful on restarts)')
  cmd:option('-batchSize', batchsize, 'mini-batch size (1 = pure stochastic)')
  cmd:option('-test_batchSize', test_batchsize, 'test mini-batch size')
  cmd:option('-test_initialization', test_initialization, 'test_initalization')

  cmd:option('-LR', 0.2, 'learning rate; if set, overrides default LR/WD recipe')
  cmd:option('-momentum', 0.9,  'momentum')
  cmd:option('-weightDecay', 0.00000, 'weight decay')

  cmd:option('-use_stn', false, '')
  cmd:option('-sampling_grid_size', sampleSize[2], '')
  cmd:option('-netType', network, 'Options: alexnet | overfeat')
  cmd:option('-retrain', initial_model, 'provide path to model to retrain with')

  cmd:option('-optimState', initial_optimState, 'provide path to an optimState to reload from')

  cmd:option('-display', 10, 'interval for printing train loss per minibatch')
  cmd:option('-snapshot', 12000, 'interval for conditional_save')
  cmd:text()

  local opt = cmd:parse(arg or {})
  opt.sampleSize = sampleSize
  opt.loadSize = loadSize
  opt.nGPU = nGPU
  -- add commandline specified options
  opt.save = paths.concat(opt.cache, 
    cmd:string(network, opt, {retrain=true, optimState=true, cache=true, data=true}))
  opt.save = paths.concat(opt.save, exp_name .. os.date():gsub(' ',''))

  print('===> Saving everything to: ' .. opt.save)
  os.execute('mkdir -p ' .. opt.save)

  return opt
end

-- return nil initially
return M

