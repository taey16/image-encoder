

function test(epoch)
  model:evaluate()
  local start = os.clock()

  local loss= 0
  local err = 0
  local batchsize = opt.test_batchsize
  local num_batches = datasetVal:getNumBatches()

  for batchidx = 1, num_batches do
    local batch_loss, batch_err = test_batch(batchidx)
    loss = loss + batch_loss 
    err = err + batch_err
  end

  local total_samples= num_batches * batchsize
  local averaged_loss= loss/ num_batches
  local averaged_err = err / total_samples * 100
  local current_time = os.clock()
  local elapsed = current_time - start

  logger_test:add{
    ['time'] = current_time,
    ['elapsed']= elapsed,
    ['epoch']= epoch,
    ['loss'] = averaged_loss,
    ['err']  = averaged_err 
  }
  print(string.format(
    'epoch: %05d val loss: %.6f err: %.6f in %.4f', 
    epoch, averaged_loss, averaged_err, elapsed )
  )
  io.flush(print(''))

end -- end of test

