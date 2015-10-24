
require 'optim'

local optimState = {
  learningRate = opt.LR,
  learningRateDecay = 0.0,
  momentum = opt.momentum,
  dampening = 0.0,
  weightDecay = opt.weightDecay
}
if opt.optimState then
  assert(paths.filep(opt.optimState), 'File not found: ' .. opt.optimState)
  print('Loading optimState from file: ' .. opt.optimState)
  optimState = torch.load(opt.optimState)
end

local optimizer = nn.Optim(model, optimState)

local function paramsForEpoch(epoch)
  local regimes = {
      -- start, end,    LR,   WD,
      {  1,      9,   opt.LR,   0.00002 },
      { 10,     18,   opt.LR/2,  0.00002 },
      { 19,     26,   opt.LR/2/2, 0.00002 },
      { 27,     36,   opt.LR/2/2/2,0.00002 },
      { 37,     50,   opt.LR/2/2/2/2,   0 },
      { 50,     60,   opt.LR/2/2/2/2/2,   0 },
      { 61,     70,   opt.LR/2/2/2/2/2/2, 0},
      { 71,     1e-8, opt.LR/2/2/2/2/2/2/2, 0},
  }
  for _, row in ipairs(regimes) do
    if epoch >= row[1] and epoch <= row[2] then
      return { learningRate=row[3], weightDecay=row[4] }, epoch == row[1]
    end
  end
end -- end of parseForEpoch

-- Create loggers.
trainLogger = optim.Logger(paths.concat(opt.save, 'train.log'))
local batchNumber
local top1_epoch, loss_epoch

-- train
function train()

  local params, newRegime = paramsForEpoch(epoch)
  optimizer:setParameters(params)
  if newRegime then
    -- Zero the momentum vector by throwing away previous state.
    optimizer = nn.Optim(model, optimState)
  end
  -- reset batchNumber
  batchNumber = 0
  cutorch.synchronize()

  -- set the dropouts to training mode
  model:training()
  model:cuda()

  local tm = torch.Timer()
  top1_epoch = 0
  loss_epoch = 0
  -- Threading main
  for i=1,opt.epochSize do
    -- queue jobs to data-workers
    -- donkey is decleared in data.lua:13 with Threads()
    donkeys:addjob(
      -- the job callback (runs in data-worker thread)
      function()
        local inputs, labels = trainLoader:sample(opt.batchSize)
        return sendTensor(inputs), sendTensor(labels)
      end,
      -- the end callback (runs in the main thread)
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
    ['error']= top1_epoch,
  }
  print(('epoch: %d trn loss: %.6f err: %.6f elapsed: %.4f'):format(
    epoch, loss_epoch, top1_epoch, elapsed))

  collectgarbage()

end -- of train()
-------------------------------------------------------------------------------------------

function conditional_save()
  -- clear the intermediate states in the model before saving to disk
  -- this saves lots of disk space
  local function sanitize(net)
    local list = net:listModules()
    for _,val in ipairs(list) do
      for name,field in pairs(val) do
        if torch.type(field) == 'table' then break end
        if torch.type(field) == 'cdata' then val[name] = nil end
        if name == 'homeGradBuffers' then val[name] = nil end
        if name == 'input_gpu' then val['input_gpu'] = {} end
        if name == 'gradOutput_gpu' then val['gradOutput_gpu'] = {} end
        if name == 'gradInput_gpu' then val['gradInput_gpu'] = {} end
        if (name == 'output' or name == 'gradInput') then
          val[name] = field.new()
        end
      end
    end
  end
  sanitize(model)
  local dump_model_path = paths.concat(opt.save, 'model_' .. epoch .. '.t7')
  local dump_optimState_path = paths.concat(opt.save, 'optimState_' .. epoch .. '.t7')
  torch.save(dump_model_path, model)
  torch.save(dump_optimState_path, optimState)
  print('Dump ' .. dump_model_path)
  print('Dump ' .. dump_optimState_path)
end

-- create tensor buffers in main thread and deallocate their storages.
-- the thread loaders will push their storages to these buffers when done loading
local inputsCPU = torch.FloatTensor()
local labelsCPU = torch.LongTensor()

-- GPU inputs (preallocate)
local inputs = torch.CudaTensor()
local labels = torch.CudaTensor()

local timer = torch.Timer()
local dataTimer = torch.Timer()

-- 4. trainBatch - Used by train() to train a single batch after the data is loaded.
function trainBatch(inputsThread, labelsThread)
  cutorch.synchronize()
  collectgarbage()

  local dataLoadingTime = dataTimer:time().real
  timer:reset()
  -- set the data and labels to 
  -- the main thread tensor buffers (free any existing storage)
  receiveTensor(inputsThread, inputsCPU)
  receiveTensor(labelsThread, labelsCPU)

  -- transfer over to GPU
  inputs:resize(inputsCPU:size()):copy(inputsCPU)
  labels:resize(labelsCPU:size()):copy(labelsCPU)

  local loss, outputs = optimizer:optimize(optim.sgd, inputs, labels, criterion)
  cutorch.synchronize()

  batchNumber = batchNumber + 1
  loss_epoch = loss_epoch + loss 

  local outputsCPU = outputs:float()
  local _, preds = outputsCPU:max(2)
  local err = (opt.batchSize - preds:eq(labelsCPU):sum())
  local top1= err / opt.batchSize * 100
  top1_epoch= top1_epoch + err

  if batchNumber % opt.display == 0 then
    io.flush(print(
      ('%04d/%04d loss %.6f err: %03.4f lr: %.6f elapsed: %.4f(%.3f)'):format( 
      batchNumber, opt.epochSize, loss, top1, optimState.learningRate, timer:time().real, dataLoadingTime)))
  end
  if batchNumber % opt.snapshot == 0 then
    conditional_save()
  end

  dataTimer:reset()

end -- end of trainBatch
