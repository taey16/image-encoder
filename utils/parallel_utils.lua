require 'cutorch'
require 'cunn'

function makeDataParallel(model, nGPU, primary_gpu_id)
  if #nGPU > 1 then
    print('converting module to nn.DataParallelTable')
    assert(#nGPU <= cutorch.getDeviceCount(), 
      'number of GPUs less than nGPU specified')
    local model_single = model
    -- Split along first (batch) dimension
    -- in our case it is batch_size
    model = nn.DataParallelTable(1)
    for _,gpu_id in pairs(nGPU) do
      cutorch.setDevice(gpu_id)
      print('DataParallelTable setDevice: '..gpu_id)
      model:add(model_single:clone():cuda(), gpu_id)
    end
    -- set 'primary' GPU
    cutorch.setDevice(primary_gpu_id)
  end
  return model
end


function splitDataParallelTable(model)
  local feature = model:get(1):get(1):clone()
  local classifier = model:get(2):clone()

  return feature, classifier
end


