
require 'optim'
paths.dofile('utils/net_utils.lua')
paths.dofile('utils/util.lua')
paths.dofile('utils/image_utils.lua')

local optimState = {
  learningRate = opt.LR,
  learningRateDecay = 0.0, 
  momentum = opt.momentum,
  dampening = 0.0,
  weightDecay = opt.weightDecay,
  -- ADAM: A Method for Stochastic Optimization, ICLR, 2015
  -- ADAM i.e. ADAptive Moment estimate
  -- Good default settings for the tested machine learning problems
  -- recommended setting: learningRate: 0.001, beta_1: 0.9, beta_2: 0.999, epsilon: 10e-8 where
  -- learningRate: stepsize
  -- \beta_1, \beta_2 \in [0, 1): exponential decay rates for the moment estimates
  beta1 = 0.9,
  beta2 = 0.999,
  epsilon = 1e-8
}
if opt.optimState then
  assert(paths.filep(opt.optimState), 
    'File not found: ' .. opt.optimState)
  print('===> Loading optimState from file: ' .. opt.optimState)
  optimState = torch.load(opt.optimState)
  print('optimState.learningRate: '..optimState.learningRate)
  print('optimState.momentum: '..optimState.momentum)
  print('optimState.weightDecay: '..optimState.weightDecay)
end


trainLogger = optim.Logger(paths.concat(opt.save, 'train.log'))
local iter_batch = 0
local top1_epoch, loss_epoch


function train()
  cutorch.synchronize()

  model:training()

  local tm = torch.Timer()
  top1_epoch = 0
  loss_epoch = 0
  for iter = 1,opt.epochSize do
    donkeys:addjob(
      function()
        local  inputs, labels = trainLoader:sample(opt.batchSize)
        return sendTensor(inputs), sendTensor(labels)
      end,
      trainBatch
    )
  end

  donkeys:synchronize()
  cutorch.synchronize()

  top1_epoch = top1_epoch / (opt.batchSize * opt.epochSize) * 100
  loss_epoch = loss_epoch / opt.epochSize

  local elapsed = tm:time().real
  trainLogger:add{
    ['time'] = elapsed,
    ['epoch']= epoch,
    ['loss'] = loss_epoch,
    ['err']= top1_epoch,
  }
  print(('epoch: %d trn loss: %.6f err: %.6f solver: %s, elapsed: %.4f'):format(
    epoch, loss_epoch, top1_epoch, opt.solver, elapsed))

  conditional_save(model, optimState, epoch)
  collectgarbage()

end -- of train()


local inputsCPU = torch.FloatTensor()
local labelsCPU = torch.LongTensor()
local inputs = torch.CudaTensor()
local labels = torch.CudaTensor()

local timer = torch.Timer()
local dataTimer = torch.Timer()


local parameters, gradParameters = model:getParameters()
function trainBatch(inputsThread, labelsThread)
  cutorch.synchronize()
  collectgarbage()
  local elapsed_batch_loading = dataTimer:time().real
  timer:reset()

  -- decay the learning rate for both LM and CNN
  local learning_rate = optimState.learningRate
  if iter_batch > opt.learning_rate_decay_start and opt.learning_rate_decay_start >= 0 then
    local frac = (iter_batch - opt.learning_rate_decay_start) / opt.learning_rate_decay_every
    local decay_factor = math.pow(0.5, frac)
    optimState.learningRate = learning_rate * decay_factor
  end

  receiveTensor(inputsThread, inputsCPU)
  receiveTensor(labelsThread, labelsCPU)
  inputs:resize(inputsCPU:size()):copy(inputsCPU)
  labels:resize(labelsCPU:size()):copy(labelsCPU)

  local loss, outputs
  feval = function(x)
    model:zeroGradParameters()
    outputs = model:forward(inputs)
    loss = criterion:forward(outputs, labels)
    local gradOutputs = criterion:backward(outputs, labels)
    model:backward(inputs, gradOutputs)
    return loss, gradParameters
  end

  if opt.solver == 'sgd' then
    optim.sgd(feval, parameters, optimState)
  elseif opt.solver == 'nag' then
    optim.nag(feval, parameters, optimState)
  elseif opt.solver == 'adam' then
    optim.adam(feval, parameters, optimState)
  else
    io.flush(print(string.format('Unknown solver: %s', opt.solver)))
    opt.solver = 'nag'
    io.flush(print(string.format('default solver: %s', opt.solver)))
    optim.nag(feval, parameters, optimState)
  end

  iter_batch = iter_batch + 1
  loss_epoch = loss_epoch + loss 

  local outputsCPU = outputs:float()
  local _, preds = outputsCPU:max(2)
  local err = (opt.batchSize - preds:eq(labelsCPU):sum())
  local top1= err / opt.batchSize * 100
  top1_epoch= top1_epoch + err


  if iter_batch % opt.display == 0 then
    local elapsed_batch = timer:time().real
    local elapsed_whole = elapsed_batch + elapsed_batch_loading
    local time_left = (opt.epochSize - (iter_batch % opt.epochSize)) * elapsed_whole
    io.flush(print(
      ('%04d/%04d %.2f loss %.6f err: %03.4f lr: %.8f wc: %.8f solver: %s, elapsed: %.4f(%.3f), time-left: %.2f hr.'):format( 
      iter_batch, opt.epochSize, iter_batch / opt.epochSize, 
      loss, top1, 
      optimState.learningRate, optimState.weightDecay, opt.solver,
      elapsed_batch, elapsed_batch_loading, time_left / 3600 )))
  end
  optimState.learningRate = learning_rate
  if iter_batch % opt.snapshot == 0 then
    conditional_save(model, optimState, epoch)
  end

  dataTimer:reset()

end -- end of trainBatch
