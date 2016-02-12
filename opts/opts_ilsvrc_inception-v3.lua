
local M = {}

function M.parse(arg)
  local defaultDir= paths.concat('/data2/ImageNet/ILSVRC2012/')
  local cache_dir = paths.concat(defaultDir, 'torch_cache');
  local data_dir  = paths.concat(defaultDir, './')
  local data_shard = false
  local batchsize = 32
  local test_batchsize = 25 -- 32
  local total_train_samples = 1281167 - 1
  local network = 'inception-v3-2015-12-05'
  local loadSize  = {3, 342, 342}
  local sampleSize= {3, 299, 299}
  local nGPU = {1, 2}
  local current_epoch = 1
  local test_initialization = false
  local experiment_id = 'digits_gpu2_inception-v3-2015-12-05_adam'
  local nClasses = 1000
  local retrain_path = 
    '/storage/ImageNet/ILSVRC2012/model/inception-v3-2015-12-05/'
  if retrain_path then
    initial_model = 
      paths.concat(retrain_path, 'inception-v3-2015-12-05.cudnn.t7')
      --paths.concat(retrain_path, ('model_%d.t7'):format(current_epoch-1)) 
    initial_optimState = 
      false
      --paths.concat(retrain_path, ('optimState_%d.t7'):format(current_epoch-1))
  else
    initial_model = false
    initial_optimState = false
  end
  local solver = 'adam'
  local LR = 0.45--0.045
  local regimes = {
    -- start, end,    LR,   WD,
    {  1,      4*1,   LR, 0.00002 },
    {  4*1+1, 14*2,   LR*0.1, 0.00002 },
    { 14*2+1, 14*3,   LR*0.1*0.1, 0.00005 },
    { 14*3+1, 14*4,   LR*0.1*0.1*0.1, 0.00005 },
    { 14*4+1, 14*5,   LR*0.1*0.1*0.1*0.1, 0.00005 },
    { 14*5+1, 14*6,   LR*0.1*0.1*0.1*0.1*0.1, 0.00005 },
    { 14*6+1, 14*7,   LR*0.1*0.1*0.1*0.1*0.1*0.1, 0 },
    { 14*7+1,  200,   LR*0.1*0.1*0.1*0.1*0.1*0.1*0.1, 0 },
  }

  local cmd = torch.CmdLine()
  cmd:text()
  cmd:text('Options:')

  cmd:option('-cache', cache_dir, 'subdirectory in which to save/log experiments')
  cmd:option('-data', data_dir, 'root of dataset')
  cmd:option('-nClasses', nClasses, '# of classes')
  cmd:option('-data_shard', data_shard, 'data shard')
  cmd:option('-nDonkeys', 3, 'number of donkeys to initialize (data loading threads)')
  cmd:option('-manualSeed', 3729, 'Manually set RNG seed')

  cmd:option('-GPU', nGPU[1], 'Default preferred GPU')

  cmd:option('-nEpochs', 200, 'Number of total epochs to run')
  cmd:option('-epochSize', math.ceil(total_train_samples/batchsize), 'Number of batches per epoch')
  cmd:option('-epochNumber', current_epoch,'Manual epoch number (useful on restarts)')
  cmd:option('-batchSize', batchsize, 'mini-batch size (1 = pure stochastic)')
  cmd:option('-test_batchSize', test_batchsize, 'test mini-batch size')

  cmd:option('-solver', solver, 'nag | adam | sgd')
  cmd:option('-LR', LR, 'learning rate; if set, overrides default LR/WD recipe')
  cmd:option('-momentum', 0.9,  'momentum')
  cmd:option('-weightDecay', 0.0002, 'weight decay')

  cmd:option('-netType', network, 'Options: alexnet | overfeat')
  cmd:option('-use_stn', false, '')
  cmd:option('-sampling_grid_size', sampleSize[2], 'sampling grid size')

  cmd:option('-retrain', initial_model, 'provide path to model to retrain with')
  cmd:option('-optimState', initial_optimState, 'provide path to an optimState to reload from')

  cmd:option('-test_initialization', test_initialization, 'test_initialization')
  cmd:option('-display', 5, 'interval for printing train loss per minibatch')
  cmd:option('-snapshot', 20000, 'interval for conditional_save')
  cmd:text()

  local opt = cmd:parse(arg or {})
  opt.loadSize = loadSize
  opt.sampleSize= sampleSize
  opt.nGPU = nGPU
  opt.regimes = regimes

  opt.save = paths.concat(opt.cache, 
    cmd:string(network, opt, {retrain=true, optimState=true, cache=true, data=true}))
  opt.save = paths.concat(opt.save, experiment_id..'_'..os.date():gsub(' ','_'):gsub(':','_'))

  print('===> Saving everything to: ' .. opt.save)
  os.execute('mkdir -p ' .. opt.save)

  return opt
end

return M

