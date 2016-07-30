
local dataset_root = '/data2/ImageNet/ILSVRC2012/'
local checkpoint_path = paths.concat(dataset_root, 'torch_cache');

local dataset_name = 'ILSVRC2012'
local total_train_samples = 1281167 - 1 -- 1 image cannot be loaded using graphicsmagick(ref. https://groups.google.com/forum/#!topic/torch7/fOSTXHIESSU)
local nClasses = 1000
local forceClasses = nil -- in case you need to set manual class-id (default: nil, class-ids are set automatically in dataset.lua and donkey/donkey.lua)

local network = 
  'resception'
local loadSize  = {3, 342, 342} -- loading size prior to crop-jittering
local sampleSize= {3, 299, 299} -- actual image size
local nGPU = {1,2} -- if you use GPU-0 and GPU-1 (e.g. CUDA_VISIBLE_DEVICES=0,1 luajit(th) main.lua
local current_epoch = 1 -- set starting epoch (it can be changed if you retrain your nets)
local current_iter = 0 -- set starting # of iterations (it needs to be used for annealing learning rates)
local test_initialization = false -- perform test initially if needed
local retrain_path = 
  false -- specify trained model paths for retraining
if retrain_path then
  initial_model = 
    paths.concat(retrain_path, ('model_%d.t7'):format(current_epoch-1)) 
  initial_optimState = 
    paths.concat(retrain_path, ('optimState_%d.t7'):format(current_epoch-1))
else
  initial_model = false
  initial_optimState = false
end

local batchsize = 32
local test_batchsize = 25 -- due to # of val. images(i.e. 50000) is divided by 25
local solver = 'nag' -- perform better then adam in our experiments
local num_max_epoch = 500
local learning_rate = 0.045 -- following google's paper (Rethinking bla~bla~)
local weight_decay = 0.00002 -- we think weight-decay(L2-regularizer) can be set small because of using BN
local learning_rate_decay_seed = 0.94 -- following google;s paper (Rethinking bla~bla~)
local learning_rate_decay_start = 0 -- starting learning rate decaying(annealing) for the first iteration
local learning_rate_decay_every = 40037 * 2 -- reduce learning rate every 2 epoch(40037*2 iterations) factor of 0.94
local grad_clip = -1 -- clip gradient
local experiment_id = string.format(
  '%s_default_opts_gpu%d_cudnn-v5_%s_epoch%d_%s_lr%.5f_clip%.3f_decay_seed%.3f_start%d_every%d', 
  dataset_name, 
  #nGPU, 
  network, 
  current_epoch, 
  solver, 
  learning_rate, grad_clip, learning_rate_decay_seed, learning_rate_decay_start, learning_rate_decay_every
)

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
cmd:option('-nEpochs', num_max_epoch, 'Number of total epochs to run')
cmd:option('-epochSize', math.ceil(total_train_samples/batchsize), 'Number of batches per epoch')
cmd:option('-epochNumber', current_epoch,'Manual epoch number (useful on restarts)')
cmd:option('-iter_batch', current_iter,'Manual iter number (useful on restarts and lr anneal)')
cmd:option('-batchSize', batchsize, 'mini-batch size (1 = pure stochastic)')
cmd:option('-test_batchSize', test_batchsize, 'test mini-batch size')
cmd:option('-test_initialization', test_initialization, 'test_initialization')
cmd:option('-retrain', initial_model, 'provide path to model to retrain with')
cmd:option('-optimState', initial_optimState, 'provide path to an optimState to reload from')

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
cmd:option('-grad_clip', grad_clip, 'gradient cliping(-1 = dont)')

-- network specific
cmd:option('-netType', network, 'Options: [inception_v3 | resception]')

-- misc.
cmd:option('-nDonkeys', 4, 'number of donkeys to initialize (data loading threads)')
cmd:option('-manualSeed', 999, 'Manually set RNG seed')
cmd:option('-display', 5, 'interval for printing train loss per minibatch')
cmd:option('-snapshot', 25000, 'interval for conditional_save')
cmd:text()

local opt = cmd:parse(arg)
opt.loadSize = loadSize
opt.sampleSize= sampleSize
opt.nGPU = nGPU

if not os.execute('cd ' .. opt.data) then
  error(("could not chdir to '%s'"):format(opt.data))
end

opt.save = paths.concat(opt.cache, experiment_id)
os.execute('mkdir -p ' ..opt.save)
print('===> checkpoint path: ' .. opt.save)

return opt

