
local M = {}

function M.parse(arg)
  local defaultDir= paths.concat('/storage/product/det/')
  local cache_dir = paths.concat(defaultDir, 'torch_cache');
  local data_dir  = paths.concat(defaultDir, './')
  local data_shard = false
  local batchsize = 32
  local test_batchsize = 20
  local total_train_samples = 215303
  local network = 'inception6' --'vgg16caffe'
  local loadSize  = {3, 256, 256}
  local sampleSize= {3, 224, 224}
  local nGPU = {1, 2}
  local current_epoch = 1
  local test_initialization = false
  local exp_name = 'det_stn'
  local backend = 'cudnn'
  --local retrain_path = '/storage/ImageNet/ILSVRC2012/torch_cache/inception6/gpu_2_lr0.045ThuNov2612:23:162015/'
  local retrain_path = '/storage/product/det/torch_cache/inception6/detWedDec218:15:202015'
  if retrain_path then
    initial_model = paths.concat(retrain_path, 'model_10.t7') 
    --initial_optimState = paths.concat(retrain_path, 'optimState_27.t7')
    initial_optimState = false
  else
    initial_model = false
    initial_optimState = false
  end
  local LR = 0.00000045
  local regimes = {
    -- start, end,    LR,   WD,
    {  1,     12,   LR, 0.00002 },
    { 13,     24,   LR*0.1, 0.00002 },
    { 25,     35,   LR*0.1*0.1, 0.00001 },
    { 36,     45,   LR*0.1*0.1*0.1, 0.00001 },
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
  cmd:option('-nDonkeys', 4, 'number of donkeys to initialize (data loading threads)')
  cmd:option('-donkey_filename', 'donkey/donkey.lua', 'donkey file to use')

  cmd:option('-manualSeed', 1234, 'Manually set RNG seed')

  cmd:option('-GPU', nGPU[1], 'Default preferred GPU')
  cmd:option('-backend', backend, 'cudnn | cunn | nn')

  cmd:option('-nEpochs', 300, 'Number of total epochs to run')
  cmd:option('-epochSize', math.ceil(total_train_samples/batchsize), 'Number of batches per epoch')
  cmd:option('-epochNumber', current_epoch,'Manual epoch number (useful on restarts)')
  cmd:option('-batchSize', batchsize, 'mini-batch size (1 = pure stochastic)')
  cmd:option('-test_batchSize', test_batchsize, 'test mini-batch size')
  cmd:option('-test_initialization', test_initialization, 'test_initalization')

  cmd:option('-LR', LR, 'Base learning rate')
  cmd:option('-momentum', 0.9, 'momentum')
  cmd:option('-weightDecay', 0.0000, 'weight decay')

  cmd:option('-use_stn', true, 'wether to use spatial transformer or not')
  cmd:option('-sampling_grid_size', sampleSize[2], '')
  cmd:option('-netType', network, 'Network model to use')
  cmd:option('-retrain', initial_model, 'provide path to model to retrain with')
  cmd:option('-optimState', initial_optimState, 'provide path to an optimState to reload from')

  cmd:option('-display', 5, 'Intervals for printing train loss per minibatch')
  cmd:option('-snapshot', 8000, 'Intervals for conditional_save')
  cmd:text()

  local opt = cmd:parse(arg or {})
  opt.loadSize = loadSize
  opt.sampleSize = sampleSize
  opt.nGPU = nGPU
  opt.regimes = regimes

  opt.save = paths.concat(opt.cache, 
    cmd:string(network, opt, {retrain=true, optimState=true, cache=true, data=true}))
  opt.save = paths.concat(opt.save, exp_name .. os.date():gsub(' ',''))

  print('===> Saving everything to: ' .. opt.save)
  os.execute('mkdir -p ' .. opt.save)

  return opt
end

-- return nil initially
return M

