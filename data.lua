
--local ffi = require 'ffi'
local Threads = require 'threads'

do 
  if opt.nDonkeys > 0 then
    local options = opt
    donkeys = Threads(
      opt.nDonkeys,
      function()
        require 'torch'
        require 'cutorch'
      end,
      function(idx)
        opt = options
        tid = idx
        local seed = opt.manualSeed + idx
        torch.manualSeed(seed)
        local cu_seed = opt.manualSeed + 321 + idx
        cutorch.manualSeedAll(seed)
        print(
          ('===> Starting donkey with id: %d seed: %d, cu_seed: %d'):format(tid-1, seed, cu_seed))
        paths.dofile('donkey/donkey.lua')
      end
    )
  else 
    paths.dofile('donkey/donkey.lua')
    donkeys = {}
    -- f1: main callback ,f2: ending callback
    function donkeys:addjob(f1, f2) f2(f1()) end
    function donkeys:synchronize() end
  end
end

--[[
nClasses= nil
classes = nil
donkeys:addjob(
  function() 
    return trainLoader.classes 
  end, 
  function(c) 
    classes = c 
  end)
donkeys:synchronize()
nClasses = #classes
assert(nClasses, "Failed to get nClasses")
print('===> nClasses: ', nClasses)
assert(nClasses == opt.nClasses, 'nClasses is mismatched')
torch.save(paths.concat(opt.save, 'classes.t7'), classes)

nTest = 0
donkeys:addjob(
  function() 
    return testLoader:size() 
  end, 
  function(c) 
    nTest = c 
  end)
donkeys:synchronize()
assert(nTest > 0, "Failed to get nTest")
print('===> nTest: ', nTest)
--]]
