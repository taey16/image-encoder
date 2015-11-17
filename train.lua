
require 'optim'
paths.dofile('utils/net_utils.lua')
paths.dofile('utils/util.lua')

local optimState = {
  learningRate = opt.LR,
  learningRateDecay = 0.0,
  momentum = opt.momentum,
  dampening = 0.0,
  weightDecay = opt.weightDecay
}
if opt.optimState then
  assert(paths.filep(opt.optimState), 
    'File not found: ' .. opt.optimState)
  print('Loading optimState from file: ' .. opt.optimState)
  optimState = torch.load(opt.optimState)
end

local regimes = {
  -- start, end,    LR,   WD,
  {  1,      9,   opt.LR, 0.00002 },
  { 10,     18,   opt.LR/2, 0.00002 },
  { 19,     26,   opt.LR/2/2, 0.00002 },
  { 27,     36,   opt.LR/2/2/2, 0.00002 },
  { 37,     50,   opt.LR/2/2/2/2, 0 },
  { 50,     60,   opt.LR/2/2/2/2/2, 0 },
  { 61,     70,   opt.LR/2/2/2/2/2/2, 0},
  { 71,    200,   opt.LR/2/2/2/2/2/2/2, 0},
  {201,   1e-8,   opt.LR/2/2/2/2/2/2/2/2, 0},
}

trainLogger = optim.Logger(paths.concat(opt.save, 'train.log'))
local batchNumber
local top1_epoch, loss_epoch


function train()
  local params, newRegime = paramsForEpoch(regimes, epoch)
  if newRegime then
    optimState = reset_optimState(params)
  end
  -- reset batchNumber
  batchNumber = 0
  cutorch.synchronize()

  model:training()

  local tm = torch.Timer()
  top1_epoch = 0
  loss_epoch = 0
  for i=1,opt.epochSize do
    donkeys:addjob(
      function()
        local  inputs, labels = trainLoader:sample(opt.batchSize)
        return sendTensor(inputs), sendTensor(labels)
        --return inputs, labels
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
  print(('epoch: %d trn loss: %.6f err: %.6f elapsed: %.4f'):format(
    epoch, loss_epoch, top1_epoch, elapsed))

  collectgarbage()

end -- of train()

local inputsCPU = torch.FloatTensor()
local labelsCPU = torch.LongTensor()
local inputs = torch.CudaTensor()
local labels = torch.CudaTensor()

local timer = torch.Timer()
local dataTimer = torch.Timer()


local parameters, gradParameters = model:getParameters()
--function trainBatch(inputsCPU, labelsCPU)
function trainBatch(inputsThread, labelsThread)
  cutorch.synchronize()
  collectgarbage()
  local dataLoadingTime = dataTimer:time().real
  timer:reset()

  receiveTensor(inputsThread, inputsCPU)
  receiveTensor(labelsThread, labelsCPU)
  -- transfer over to GPU
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
  optim.sgd(feval, parameters, optimState)
  --optim.adagrad(feval, parameters, optimState)

  -- DataParallelTable's syncParameters
  model:apply(
    function(m) 
      if m.syncParameters then m:syncParameters() end 
    end
  )
  cutorch.synchronize()

  batchNumber= batchNumber + 1
  loss_epoch = loss_epoch + loss 

  local outputsCPU = outputs:float()
  local _, preds = outputsCPU:max(2)
  local err = (opt.batchSize - preds:eq(labelsCPU):sum())
  local top1= err / opt.batchSize * 100
  top1_epoch= top1_epoch + err

  if batchNumber % opt.display == 0 then
    local elapsed_batch = timer:time().real
    local elapsed_whole = elapsed_batch + dataLoadingTime
    local time_left = (opt.epochSize - batchNumber) * elapsed_whole
    io.flush(print(
      ('%04d/%04d loss %.6f err: %03.4f lr: %.6f elapsed: %.4f(%.3f), time-left: %.2f hr.'):format( 
      batchNumber, opt.epochSize, loss, top1, 
      optimState.learningRate, 
      elapsed_batch, dataLoadingTime, time_left / 3600 )))
  end
  if batchNumber % opt.snapshot == 0 then
    conditional_save(model, optimState, epoch)
  end

  dataTimer:reset()

end -- end of trainBatch
