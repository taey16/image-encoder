
local M = {}

function M.parse(arg)
  local defaultDir= paths.concat('/storage/product/det/')
  local cache_dir = paths.concat(defaultDir, 'torch_cache');
  local data_dir  = paths.concat(defaultDir, './')
  local batchsize = 20
  local total_train_samples = 215303
  local network = 'inception6' --'vgg16caffe'
  local sampleSize= {3, 224, 224}
  local loadSize  = {3, 256, 256}
  
  local backend = 'cudnn'
  local donkey_filename = 'donkey_det.lua'
  --[[
  local retrain_path = '/storage/ImageNet/ILSVRC2012/torch_cache/inception6/stn_TueOct2023:01:282015/'
  if retrain_path then
    initial_model = paths.concat(retrain_path, 'model_27.t7') 
    initial_optimState = paths.concat(retrain_path, 'optimState_27.t7')
  else
    initial_model = nil
    initial_optimState = nil
  end
  --]]
  local retrain_path = '/storage/product/det/torch_cache/inception6/stn_epoch28/'
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
  cmd:option('-donkey_filename', ('donkey/%s'):format(donkey_filename), 'donkey file to use')

  cmd:option('-manualSeed', 1234, 'Manually set RNG seed')

  cmd:option('-GPU', 1, 'Default preferred GPU')
  cmd:option('-nGPU', 1, 'Number of GPUs to use by default')
  cmd:option('-backend', backend, 'cudnn | cunn | nn')

  cmd:option('-nEpochs', 300, 'Number of total epochs to run')
  cmd:option('-epochSize', math.ceil(total_train_samples/batchsize), 'Number of batches per epoch')
  cmd:option('-epochNumber', 28,'Manual epoch number (useful on restarts)')

  cmd:option('-batchSize', batchsize, 'mini-batch size')
  cmd:option('-test_batchSize', batchsize, 'test mini-batch size')
  cmd:option('-test_initialization', true, 'test_initalization')

  cmd:option('-LR', 0.002, 'Base learning rate')
  cmd:option('-momentum', 0.9, 'momentum')
  cmd:option('-weightDecay', 0.0000, 'weight decay')

  cmd:option('-sampleSize', sampleSize, 'Size of cropped region')
  cmd:option('-loadSize', loadSize, 'Size of original input')
  cmd:option('-netType', network, 'Network model to use')
  cmd:option('-retrain', initial_model, 'provide path to model to retrain with')
  cmd:option('-optimState', initial_optimState, 'provide path to an optimState to reload from')
  cmd:option('-use_stn', true, 'wether to use spatial transformer or not')

  cmd:option('-display', 10, 'Intervals for printing train loss per minibatch')
  cmd:option('-snapshot', 4000, 'Intervals for conditional_save')
  cmd:text()

  local opt = cmd:parse(arg or {})
  opt.save = paths.concat(opt.cache, 
    cmd:string(network, opt, {retrain=true, optimState=true, cache=true, data=true}))
  opt.save = paths.concat(opt.save, 'stn_epoch28' )

  print('===> Saving everything to: ' .. opt.save)
  os.execute('mkdir -p ' .. opt.save)

  return opt
end

-- return nil initially
return M

