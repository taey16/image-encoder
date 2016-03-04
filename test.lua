
paths.dofile('utils/util.lua')

local testLogger = optim.Logger(paths.concat(opt.save, 'test.log'))

local iter_batch
local error_for_all_batch
local loss_for_all_batch
local num_samples_total
donkeys:addjob(
  function() return testLoader:size() end,
  function(input) num_samples_total = input end
)
donkeys:synchronize()
local num_batches = math.floor(num_samples_total*1.0/opt.test_batchSize)
local num_samples = num_batches * opt.test_batchSize

local timer = torch.Timer()
function test()
  cutorch.synchronize()
  protos.encoder:evaluate()
  protos.classifier:evaluate()
  timer:reset()

  iter_batch = 0
  error_for_all_batch = 0
  loss_for_all_batch = 0
  for i=1,num_batches do 
    local indexStart= (i-1) * opt.test_batchSize + 1
    local indexEnd = (indexStart + opt.test_batchSize - 1)
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

  error_for_all_batch= error_for_all_batch * 100 / num_samples
  loss_for_all_batch = loss_for_all_batch / num_batches

  local elapsed = timer:time().real
  testLogger:add{
    ['time'] = elapsed, 
    ['epoch'] = epoch,
    ['loss'] = loss_for_all_batch,
    ['err'] = error_for_all_batch,
  }
  print(('epoch: %d tst loss: %.6f err: %.6f elapsed: %.4f\n'):format(
    epoch, loss_for_all_batch, error_for_all_batch, timer:time().real))

end -- of test()


local inputsCPU = torch.FloatTensor()
local labelsCPU = torch.LongTensor()
local inputs = torch.CudaTensor()
local labels = torch.CudaTensor()


function testBatch(inputsThread, labelsThread)
  receiveTensor(inputsThread, inputsCPU)
  receiveTensor(labelsThread, labelsCPU)
  inputs:resize(inputsCPU:size()):copy(inputsCPU)
  labels:resize(labelsCPU:size()):copy(labelsCPU)

  local feat = protos.encoder:forward(inputs)
  local outputs = protos.classifier:forward(feat)
  local loss_batch = criterion:forward(outputs, labels)
  cutorch.synchronize()

  iter_batch = iter_batch + opt.test_batchSize
  loss_for_all_batch = loss_for_all_batch + loss_batch
  local _, preds = outputs:max(2)
  local err = opt.test_batchSize - preds:eq(labels):sum()
  error_for_all_batch = error_for_all_batch + err

  if iter_batch % (opt.display*4) == 0 then
    io.flush(print(('%04d loss: %.6f err: %.6f'):format(
      iter_batch, loss_batch , err / opt.test_batchSize)))
  end

end -- end of testBatch

