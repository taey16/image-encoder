
function test_batch(batchidx)
  local inputs, labels = datasetVal:getBatch(batchidx)
  if opt.use_cuda then
    inputs = inputs:cuda()
    labels = labels:cuda()
  end
  local pred = model:forward(inputs)
  local loss = criterion:forward(pred, labels)
  local _, preds = pred:max(2)
  local err = opt.test_batchsize - preds:eq(labels):sum()

  return loss, err
end
