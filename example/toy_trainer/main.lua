
require 'image'

opt = {
  gpu_id = 1, 
  nGPU = 2,
  use_cuda = true,
  backend = 'cudnn',
  do_distort = true,
  do_yuv_lcn = false,
  use_stn = false,
  sampling_grid_size = 32,
  model_script = 'models/nin.lua',
  model_filename = nil,
  data_loader_file = 'data_provider/distort_mnist.lua',
  batchsize = 256,
  test_batchsize = 100,
  test_initialization = false,
  test_interval = 1,
  max_epoch = 4000,
  base_lr = 1.0,
  lr_decay= 0.0, --1e-7,
  momentum = 0.9,
  optimState = nil,
  epoch_number = 1,
  display = 40,
  rand_seed = 777,
  num_threads = 4,
  tensortype = 'torch.FloatTensor',
  dataset_root = '/home/taey16/storage/mnist/',
  save = '/home/taey16/storage/mnist/model/nin_bn/',
  snapshot = 1,
  snapshot_prefix = 'nin_',
}
regimes = {
 -- start, end,    LR,   WD,
 {  1,     300,   opt.base_lr,    0.00002 },
 { 301,    600,   opt.base_lr/2,   0.00002 },
 { 601,    800,   opt.base_lr/2/2,  0.00002 },
 { 801,   1000,   opt.base_lr/2/2/2, 0.00002 },
 {1001,    1e8,   opt.base_lr/2/2/2/2,0.00001 },
}
opt.regimes = regimes

print(string.format('===> Save to: %s/%s', opt.save, opt.snapshot_prefix))

if opt.use_cuda then
  require 'cunn'
  require 'cudnn'
  require 'cutorch'
else
  require 'nn'
end
print('===> use_cuda: ' .. tostring(opt.use_cuda))

torch.setnumthreads(opt.num_threads)
torch.manualSeed(opt.rand_seed)
torch.setdefaulttensortype(opt.tensortype)

paths.dofile('get_data.lua')
paths.dofile('get_model.lua')
paths.dofile('get_optimizer.lua')
paths.dofile('conditional_save.lua')
paths.dofile('train.lua')
paths.dofile('test.lua')


print('===> use_stn: ' .. tostring(opt.use_stn))
print('===> do_distort: ' .. tostring(opt.do_distort))
print('===> model_script: ' .. opt.model_script)
print('===> sampling_grid_size: ' .. opt.sampling_grid_size)
print('===> test_initialization: ' .. tostring(opt.test_initialization))

paths.dofile('train_batch.lua')
paths.dofile('test_batch.lua')
epoch = opt.epoch_number
global_start = os.clock()
if opt.test_initialization then test(epoch) end
epoch = epoch + 1
while true do
  train(epoch) 
  if epoch % opt.test_interval == 0 then 
    test(epoch)
  end
  if epoch % opt.snapshot == 0 then
    conditional_save()
  end
  epoch = epoch + 1
  collectgarbage()
end

  --[[
  if opt.use_stn then
    os.execute("mkdir -p "..'locnet')
    img = spanet.output[1]
    image.save(string.format('locnet/epoch_%02d_spanet.output.png', epoch), img)
    img = tranet:get(1).output[1]
    image.save(string.format('locnet/epoch_%02d_tranet.get_1.output.png', epoch), img)
  end
  --]]
   
   --[[
   if use_stn then
      w1=image.display({image=spanet.output, nrow=16, legend='STN-transformed inputs, epoch : '..epoch, win=w1})
      w2=image.display({image=tranet:get(1).output, nrow=16, legend='Inputs, epoch : '..epoch, win=w2})
   end
   --]] 
--]]
