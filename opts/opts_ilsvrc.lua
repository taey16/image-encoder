
local M = {}

function M.parse(arg)
  local defaultDir= paths.concat('/storage/ImageNet/ILSVRC2012/')
  local cache_dir = paths.concat(defaultDir, 'torch_cache');
  local data_dir  = paths.concat(defaultDir, './')
  local data_shard = false
  local batchsize = 32
  local test_batchsize = 50
  local total_train_samples = 1281167 - 1
  local network = 'inception6' --'inception6' --'vgg16caffe'
  local loadSize  = {3, 256, 256}
  local sampleSize= {3, 224, 224}
  local nGPU = {1, 2}
  local current_epoch = 16
  local test_initialization = true
  local exp_name = 'gpu_2_lr0.045'
  local backend = 'cudnn'
  local retrain_path = '/storage/ImageNet/ILSVRC2012/torch_cache/inception6_no_bn/gpu_2_lr0.045SatNov2115:38:502015'
  if retrain_path then
    initial_model = paths.concat(retrain_path, 'model_15.t7') 
    --initial_optimState = paths.concat(retrain_path, 'optimState_15.t7')
    initial_optimState = false
  else
    initial_model = false
    initial_optimState = false
  end
  local LR = 0.045
  local regimes = {
    -- start, end,    LR,   WD,
    {  1,      8,   LR, 0.00002 },
    {  9,     16,   LR*0.1, 0.00002 },
    { 17,     24,   LR*0.1*0.1, 0.00001 },
    { 25,     32,   LR*0.1*0.1*0.1, 0.00001 },
    { 33,     40,   LR*0.1*0.1*0.1*0.1, 0 },
    { 41,     48,   LR*0.1*0.1*0.1*0.1*0.1, 0 },
    { 49,     56,   LR*0.1*0.1*0.1*0.1*0.1*0.1, 0 },
    { 57,    200,   LR*0.1*0.1*0.1*0.1*0.1*0.1*0.1, 0 },
  }

  local cmd = torch.CmdLine()
  cmd:text()
  cmd:text('Options:')

  cmd:option('-cache', cache_dir, 'subdirectory in which to save/log experiments')
  cmd:option('-data', data_dir, 'root of dataset')
  cmd:option('-data_shard', data_shard, 'data shard')
  cmd:option('-nDonkeys', 3, 'number of donkeys to initialize (data loading threads)')
  cmd:option('-donkey_filename', 'donkey/donkey.lua', 'donkey file')

  cmd:option('-manualSeed', 222, 'Manually set RNG seed')

  cmd:option('-GPU', nGPU[1], 'Default preferred GPU')
  --cmd:option('-nGPU', 1, 'Number of GPUs to use by default')
  cmd:option('-backend', backend, 'Options: cudnn | fbcunn | cunn')

  cmd:option('-nEpochs', 100, 'Number of total epochs to run')
  cmd:option('-epochSize', math.ceil(total_train_samples/batchsize), 'Number of batches per epoch')
  cmd:option('-epochNumber', current_epoch,'Manual epoch number (useful on restarts)')
  cmd:option('-batchSize', batchsize, 'mini-batch size (1 = pure stochastic)')
  cmd:option('-test_batchSize', test_batchsize, 'test mini-batch size')
  cmd:option('-test_ratio', 0.5, 'test ratio from 0 to 1')
  cmd:option('-test_initialization', test_initialization, 'test_initialization')

  cmd:option('-LR', LR, 'learning rate; if set, overrides default LR/WD recipe')
  cmd:option('-momentum', 0.9,  'momentum')
  cmd:option('-weightDecay', 0.00000, 'weight decay')

  cmd:option('-use_stn', false, '')
  cmd:option('-sampling_grid_size', sampleSize[2], '')
  cmd:option('-netType', network, 'Options: alexnet | overfeat')
  cmd:option('-retrain', initial_model, 'provide path to model to retrain with')
  cmd:option('-optimState', initial_optimState, 'provide path to an optimState to reload from')

  cmd:option('-display', 5, 'interval for printing train loss per minibatch')
  cmd:option('-snapshot', 8000, 'interval for conditional_save')
  cmd:text()

  local opt = cmd:parse(arg or {})
  opt.loadSize = loadSize
  opt.sampleSize= sampleSize
  opt.nGPU = nGPU
  opt.regimes = regimes

  -- add commandline specified options
  opt.save = paths.concat(opt.cache, cmd:string(network, opt, {retrain=true, optimState=true, cache=true, data=true}))
  opt.save = paths.concat(opt.save, exp_name .. os.date():gsub(' ',''))

  print('===> Saving everything to: ' .. opt.save)
  os.execute('mkdir -p ' .. opt.save)

  return opt
end

-- return nil initially
return M

