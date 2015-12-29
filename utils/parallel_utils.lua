require 'cutorch'


function makeDataParallel(model, nGPU, primary_gpu_id)
  if #nGPU > 1 then
    print('===> Converting module to nn.DataParallelTable')
    assert(#nGPU <= cutorch.getDeviceCount(), 
      'number of GPUs less than nGPU specified')
    local model_single = model
    -- Split along first (batch) dimension
    -- in our case it is batch_size
    model = nn.DataParallelTable(1)
    for _,gpu_id in pairs(nGPU) do
      cutorch.setDevice(gpu_id)
      print('===> DataParallelTable setDevice: '..gpu_id)
      if gpu_id == primary_gpu_id then
        model:add(model_single:cuda(), gpu_id)
      else
        model:add(model_single:clone():cuda(), gpu_id)
      end
    end
    -- set 'primary' GPU
    cutorch.setDevice(primary_gpu_id)
  end
  return model
end


function splitDataParallelTable(model)
  local feature = model:get(1):get(1)
  local classifier = model:get(2)

  return feature, classifier
end


