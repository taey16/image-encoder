
local parallel_utils = {}
function parallel_utils.makeDataParallel(model, gpus)
  --[[
  DataParallelTable(dim, [flattenParams], [useNCCL])

  Creates a DataParallelTable that splits the input on the dimension dim. 
  If flattenParams is true, getParameters() will be called on the replicated module. 
  If useNCCL is true and both NCCL and the NCCL torch bindings are installed, 
  NCCL will be used for inter-GPU communication.
  For best performance, use flattenParams and NCCL.

  DataParallelTable:add(module, gpus)

  Replicates module on the table of gpus. For example:
  nn.DataParallelTable(1):add(module, {1, 2, 3, 4})

  DataParallelTable:threads(initFunc)

  Switches the internal implementation to use a seperate thread for each replica. 
  This may hide the cost of kernel launches by dispatching them in parallel. 
  The initFunc is executed in each thread.

  nn.DataParallelTable(1):threads(function()
    require 'cudnn'
    end)

  DataParallelTable:syncParameters()

  Copies the model parameters from the first replica to all other replicas. 
  This is automatically called from updateOutput, 
  if it has not been called since the last accGradParameters.
  --]]
  local fastest, benchmark, verbose = 
    cudnn.fastest, cudnn.benchmark, cudnn.verbose
  local parallel_model = nn.DataParallelTable(1, true, true)
  parallel_model:add(model, gpus):threads(
    function()
      local cudnn = require 'cudnn'
      cudnn.fastest, cudnn.benchmark, cudnn.verbose = fastest, benchmark, verbose
    end
  )
  parallel_model.gradInput = nil
  model = parallel_model
  return model
end

return parallel_utils

