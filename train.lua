
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
}
if opt.optimState then
  assert(paths.filep(opt.optimState), 
    'File not found: ' .. opt.optimState)
  print('Loading optimState from file: ' .. opt.optimState)
  optimState = torch.load(opt.optimState)
  print('optimState.learningRate: '..optimState.learningRate)
  print('optimState.momentum: '..optimState.momentum)
  print('optimState.weightDecay: '..optimState.weightDecay)
end


trainLogger = optim.Logger(paths.concat(opt.save, 'train.log'))
local batchNumber
local top1_epoch, loss_epoch


function train()
  print('epoch: '..epoch)
  local params, newRegime = paramsForEpoch(opt.regimes, epoch)
  optimState.learningRate = params.learningRate
  optimState.weightDecay = params.weightDecay
  if newRegime then
    optimState = reset_optimState(params)
    print('reset optimState')
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
  local dataLoadingTime = dataTimer:time().real
  timer:reset()

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
  --optim.sgd(feval, parameters, optimState)
  optim.nag(feval, parameters, optimState)

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

  --[[
  if batchNumber == 1 and opt.use_stn then
    save_images(model:get(1):get(1):get(1).output:float(), opt.batchSize/2, 'save_image_'..batchNumber..'.png')
  end
  --]]

  if batchNumber % opt.display == 0 then
    local elapsed_batch = timer:time().real
    local elapsed_whole = elapsed_batch + dataLoadingTime
    local time_left = (opt.epochSize - batchNumber) * elapsed_whole
    io.flush(print(
      ('%04d/%04d loss %.6f err: %03.4f lr: %.8f elapsed: %.4f(%.3f), time-left: %.2f hr.'):format( 
      batchNumber, opt.epochSize, loss, top1, 
      optimState.learningRate, 
      elapsed_batch, dataLoadingTime, time_left / 3600 )))
    if opt.use_stn and batchNumber > 6000 then
      save_images(model:get(1):get(1):get(1).output:float(), opt.batchSize/2, 'save_image_'..batchNumber..'.png')
      --print(model:get(1):get(1):get(1):get(1):get(2):get(25).output[{{1,opt.batchSize/2},{}}]:float())
    end
  end
  if batchNumber % opt.snapshot == 0 then
    conditional_save(model, optimState, epoch)
  end

  dataTimer:reset()

end -- end of trainBatch
