
local threads = require 'threads'

local nthread = 2
local njob = 4
local msg = "hello from a satellite thread"


local pool = threads.Threads(
   nthread,
   function()
     require 'torch'
     print('0 -- init callback\n')
   end,
   function(threadid)
      print('1 -- start callback threadid ' .. threadid)
      gmsg = msg -- get it the msg upvalue and store it in thread state
   end
)

local jobdone = 0
for i=1,njob do
   pool:addjob(
      function()
         print(('2 -- main callback %s -- thread ID is %x').format(gmsg, __threadid))
         return __threadid
      end,

      function(id)
         print(string.format("3 -- end calback job %d finished (ran on thread ID %x)", i, id))
         jobdone = jobdone + 1
      end
   )
end

pool:synchronize()

print(string.format('%d jobs done', jobdone))

pool:terminate()
