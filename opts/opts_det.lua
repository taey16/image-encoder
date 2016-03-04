
local dataset_root = '/storage/product/det/'
local checkpoint_path = paths.concat(dataset_root, 'torch_cache');

local dataset_name = 'det'
local total_train_samples = 9734 + 21220 + 4964 + 18206 + 7274 + 12630 + 19567 + 20396 + 20460 + 14181 + 20048 + 19726 + 15895 + 39438 + 17304
local forceClasses = 
  {'blouse', 'cardigan', 'coat', 'hats', 'jacket', 'jumper', 'neat', 'onepiece', 'pants', 'shirt', 'shoes', 'skirt', 'swimwear', 'T-shirt', 'underwear'}
local nClasses = #forceClasses

local network = ''
local loadSize  = {3, 342, 342}
local sampleSize= {3, 299, 299}
local nGPU = {1,2}
local current_epoch = 34
local test_initialization = false
local retrain_path = 
  '/storage/ImageNet/ILSVRC2012/torch_cache/inception7_residual/digits_gpu1_inception-v3-2015-12-05_lr0.045_Mon_Jan_18_13_23_03_2016/'
if retrain_path then
  initial_model = 
    paths.concat(retrain_path, ('model_%d.t7'):format(current_epoch-1)) 
  initial_optimState = 
    false
else
  initial_model = false
  initial_optimState = false
end
current_epoch = 1

local batchsize = 32
local test_batchsize = 32
local solver = 'nag'
local num_max_epoch = 500
local learning_rate = 0.045/2/2
local weight_decay = 0.0001
local learning_rate_decay_seed = 0.5
local learning_rate_decay_start = math.floor(261043/32) * 5
local learning_rate_decay_every = math.floor(261043/32) * 5 
local experiment_id = string.format(
  '%s_X_gpu%d_%s_epoch%d_%s_lr%.5f_decay_seed%.3f_start%d_every%d', 
    dataset_name, #nGPU, network, current_epoch, solver, learning_rate, learning_rate_decay_seed, learning_rate_decay_start, learning_rate_decay_every)


cmd = torch.CmdLine()
cmd:text()
cmd:text('Train an vision encoder model')
cmd:text()
cmd:text('Options')

-- dataset specific
cmd:option('-cache', checkpoint_path, 'subdirectory in which to save/log experiments')
cmd:option('-data', dataset_root, 'root of dataset')
cmd:option('-nClasses', nClasses, '# of classes')

-- training specific
cmd:option('-nEpochs', num_max_epoch, 
  'number of total epochs to run')
cmd:option('-epochSize', math.ceil(total_train_samples/batchsize), 
  'number of batches per epoch')
cmd:option('-epochNumber', current_epoch,
  'manual epoch number (useful on restarts)')
cmd:option('-batchSize', batchsize, 
  'mini-batch size')
cmd:option('-test_batchSize', test_batchsize, 
  'test mini-batch size')
cmd:option('-test_initialization', test_initialization, 
  'test_initialization')
cmd:option('-retrain', initial_model, 
  'provide path to model to retrain with')
cmd:option('-optimState', initial_optimState, 
  'provide path to an optimState to reload from')

-- optimizer specific
cmd:option('-solver', solver, 'nag | adam | sgd')
cmd:option('-LR', learning_rate, 
  'learning rate; if set, overrides default LR/WD recipe')
cmd:option('-learning_rate_decay_seed', learning_rate_decay_seed,
  'decay_factor = math.pow(opt.learning_rate_decay_seed, frac)')
cmd:option('-learning_rate_decay_start', learning_rate_decay_start,
  'at what iteration to start decaying learning rate? (-1 = dont)')
cmd:option('-learning_rate_decay_every', learning_rate_decay_every,
  'every how many iterations thereafter to drop LR by half?')
cmd:option('-momentum', 0.9,  'momentum')
cmd:option('-weightDecay', weight_decay, 'weight decay')

-- network specific
cmd:option('-netType', network, 'Options: [inception_v3 | resception]')

-- misc.
cmd:option('-nDonkeys', 6, 'number of donkeys to initialize (data loading threads)')
cmd:option('-manualSeed', 999, 'Manually set RNG seed')
cmd:option('-display', 5, 'interval for printing train loss per minibatch')
cmd:option('-snapshot', 4000, 'interval for conditional_save')
cmd:text()

local opt = cmd:parse(arg)
opt.loadSize  = loadSize
opt.sampleSize= sampleSize
opt.nGPU = nGPU
if forceClasses then
  opt.forceClasses = forceClasses
end

if not os.execute('cd ' .. opt.data) then
  error(("could not chdir to '%s'"):format(opt.data))
end

opt.save = paths.concat(opt.cache, experiment_id)
os.execute('mkdir -p ' ..opt.save)
print('===> checkpoint path: ' .. opt.save)

return opt

