
local function paramsForEpoch(epoch, regimes)
  for _, row in ipairs(regimes) do
    if epoch >= row[1] and epoch <= row[2] then
      return { learningRate=row[3], weightDecay=row[4] }, epoch == row[1]
    end
  end
end -- end of parseForEpoch


parameters, gradParameters = model:getParameters()


function train(epoch)
  model:training()

  local params, newRegime = paramsForEpoch(epoch, opt.regimes)
  if newRegime then
    optimState = {
      learningRate = params.learningRate,
      learningRateDecay = 0.0,
      momentum = opt.momentum,
      dampening = 0.0,
      weightDecay = params.weightDecay
    }
  end

  local start = os.clock()

  local loss= 0
  local err = 0
  local batchsize = opt.batchsize
  local num_batches = datasetTrain:getNumBatches()

  for batchidx = 1, num_batches do
    local batch_loss, batch_err = train_batch(batchidx)
    loss = loss + batch_loss
    err = err + batch_err
  end

  local total_samples= (num_batches * opt.batchsize)
  local averaged_loss= loss/ num_batches
  local averaged_err = err / total_samples * 100
  local current_time = os.clock()
  local elapsed = current_time - start
  local global_elapsed = current_time - global_start

  logger_train:add {
    ['time'] = global_elapsed,
    ['elapsed'] = elapsed,
    ['lr'] = optimState.learningRate,
    ['epoch']= epoch,
    ['loss'] = averaged_loss,
    ['err']  = averaged_err 
  }
  print(string.format(
    'epoch: %05d trn loss: %.6f err: %.6f in %.4f', 
    epoch, averaged_loss, averaged_err, elapsed )
  )

end -- end of train

