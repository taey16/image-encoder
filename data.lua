
local ffi = require 'ffi'
local Threads = require 'threads'

do 
  -- start K datathreads
  if opt.nDonkeys > 0 then
    -- make an upvalue to serialize over to donkey threads
    local options = opt
    donkeys = Threads(
      opt.nDonkeys,
      function()
        require 'torch'
      end,
      function(idx)
        -- pass to all donkeys via upvalue
        opt = options
        tid = idx
        local seed = opt.manualSeed + idx
        torch.manualSeed(seed)
        if opt.data_shard == true and idx % 2 == 0 then
          opt.defaultDir = paths.concat('/data2/ImageNet/ILSVRC2012/')
          opt.cache = paths.concat(opt.defaultDir, 'torch_cache');
          opt.data = paths.concat(opt.defaultDir, './')
        end
        print(('===> Starting donkey with id: %d seed: %d'):format(tid, seed))
        paths.dofile(opt.donkey_filename)
      end
    )
  else 
    -- single threaded data loading. useful for debugging
    paths.dofile(opt.donkey_filename)
    donkeys = {}
    -- f1: main callback ,f2: ending callback
    function donkeys:addjob(f1, f2) f2(f1()) end
    function donkeys:synchronize() end
  end
end

nClasses= nil
classes = nil
donkeys:addjob(function() return trainLoader.classes end, function(c) classes = c end)
donkeys:synchronize()
nClasses = #classes
assert(nClasses, "Failed to get nClasses")
print('===> nClasses: ', nClasses)
torch.save(paths.concat(opt.save, 'classes.t7'), classes)

nTest = 0
donkeys:addjob(function() return testLoader:size() end, function(c) nTest = c end)
donkeys:synchronize()
assert(nTest > 0, "Failed to get nTest")
print('===> nTest: ', nTest)

