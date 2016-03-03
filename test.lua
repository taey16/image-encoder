
paths.dofile('utils/util.lua')

testLogger = optim.Logger(paths.concat(opt.save, 'test.log'))

--[[
local testDataIterator = function()
  testLoader:reset()
  return function() return testLoader:get_batch(false) end
end
--]]

local iter_batch
local top1_center, loss
local timer = torch.Timer()

function test()
  iter_batch = 0
  cutorch.synchronize()
  timer:reset()

  model:evaluate()

  top1_center = 0
  loss = 0
  -- nTest is set in data.lua
  local num_batches = math.floor(nTest*1.0/opt.test_batchSize)
  local num_samples_test = num_batches * opt.test_batchSize
  for i=1,num_batches do 
    local indexStart= (i-1) * opt.test_batchSize + 1
    local indexEnd  = (indexStart + opt.test_batchSize - 1)
    donkeys:addjob(
      function()
        local inputs, labels = testLoader:get(indexStart, indexEnd)
        return sendTensor(inputs), sendTensor(labels)
      end,
      testBatch
    )
  end

  donkeys:synchronize()
  cutorch.synchronize()

  top1_center = top1_center * 100 / num_samples_test
  loss = loss / num_batches
  local elapsed = timer:time().real
  testLogger:add{
    ['time'] = elapsed, 
    ['epoch'] = epoch,
    ['loss'] = loss,
    ['err'] = top1_center,
  }
  print(('epoch: %d tst loss: %.6f err: %.6f elapsed: %.4f\n'):format(
    epoch, loss, top1_center, timer:time().real))

end -- of test()

local inputsCPU = torch.FloatTensor()
local labelsCPU = torch.LongTensor()
local inputs = torch.CudaTensor()
local labels = torch.CudaTensor()

function testBatch(inputsThread, labelsThread)
  iter_batch = iter_batch + opt.test_batchSize

  receiveTensor(inputsThread, inputsCPU)
  receiveTensor(labelsThread, labelsCPU)
  inputs:resize(inputsCPU:size()):copy(inputsCPU)
  labels:resize(labelsCPU:size()):copy(labelsCPU)

  local outputs = model:forward(inputs)
  local loss_batch = criterion:forward(outputs, labels)
  cutorch.synchronize()

  loss = loss + loss_batch
  local _, preds = outputs:max(2)
  local err = opt.test_batchSize - preds:eq(labels):sum()
  top1_center = top1_center + err

  if iter_batch % (opt.display*4) == 0 then
    io.flush(print(('%04d loss: %.6f err: %.6f'):format(
      iter_batch, loss_batch , err / opt.test_batchSize)))
  end

end -- end of testBatch

