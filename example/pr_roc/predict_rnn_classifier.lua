require 'torch'
require 'nn'
require 'nngraph'
require 'cutorch'
require 'cunn'
require 'cudnn'
cudnn.benchmark = true
cudnn.fastest = true
cudnn.verbose = false
package.path = '/works/demon_11st/lua/?.lua;' .. package.path
package.path = '../../?.lua;'..package.path
local agent = require 'agent.agent_attribute'
local attribute_utils= require 'utils.attribute_utils'

attribute_id = 
  'button'
local image_list, label_list = attribute_utils.get_val(attribute_id)

for n, filename in ipairs(image_lists) do
  print(filename)
  local sents = agent.get_attribute(filename)
  print(sents)
end

io.flush(print('Done'))

