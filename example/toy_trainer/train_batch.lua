
require 'optim'

function train_batch(batchidx)

  local inputs, labels = datasetTrain:getBatch(batchidx)
  if opt.use_cuda then
    inputs = inputs:cuda()
    labels = labels:cuda()
  end

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

  -- DataParallelTable's syncParameters
  model:apply(
    function(m) 
      if m.syncParameters then m:syncParameters() end 
    end)

  local _, preds = outputs:max(2)
  local correct = preds:eq(labels):sum()
  local err = opt.batchsize - correct

  if batchidx % opt.display == 0 then 
    io.flush(print(string.format(
      'batch: %05d lr: %.6f loss: %.6f err: %.4f', 
      batchidx, optimState.learningRate, loss, err/opt.batchsize*100)))
  end

  return loss, err

end -- end of train_batch

