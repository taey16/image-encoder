
require 'torch'
local Threads  = require 'threads'


--local opts = paths.dofile('opts/opts_ilsvrc.lua')
--opt = opts.parse(arg)
--print(opt)

opt = {
  defaultDir = '/storage/ImageNet/ILSVRC2012/',
  nDonkeys = 4,
  manualSeed = 77,
  donkey_filename = 'donkey/donkey_ilsvrc12.lua',
}
opt.cache= paths.concat( opt.defaultDir , 'torch_cache')
opt.data = opt.defaultDir

paths.dofile(opt.donkey_filename)
img = testLoader.sampleHookTest(self,'/works/caffe/examples/images/cat.jpg')
print(img:type())


--[[
donkeys = Threads( 
  option.nDonkeys, 
  function() require 'torch' end,
  function(thread_index)
    local tid = thread_index
    -- should be decleared globally in function
    opt = option
    local seed= opt.manualSeed + tid
    torch.manualSeed(seed)
    print(('===> Starting donkey with id: %d seed: %d'):format(tid, seed))
    paths.dofile(opt.donkey_filename)
  end
)

nClasses = nil
donkeys:addjob(function() return trainLoader.classes end, function(c) classes = c end)
donkeys:synchronize()
nClasses = #classes
assert(nClasses, "Failed to get nClasses")
print('===> nClasses: ', nClasses)

donkeys:addjob(function() return testLoader:size() end, function(c) nTest = c end)
donkeys:synchronize()
assert(nTest > 0, "Failed to get nTest")
print('===> nTest: ', nTest)

filename = '/storage/ImageNet/ILSVRC2012/val_synset.txt'

for line in io.lines(filename) do
  donkeys:addjob(
    function()
      local sp = line:split(' ')
      -- print(sp[1] .. ' ' .. sp[2])
      return sp
    end,
    testLoader.sampleHookTest(self,sp[1])
  )
end
--]]
