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
  --'china_collar'
  --'slit_collar'
  'button'
attribute_name = 
  --'차이나카라'
  --'슬릿카라'
  '단추'
local image_list, label_list = attribute_utils.get_test(attribute_id)

local output_fp = io.open(string.format(
  'log_rnn_classifier_shuffle_cutoff100_finetune-1_%s.log.txt', attribute_id), 'w')

for n, filename in ipairs(image_list) do
  print(filename)
  local sents, prob
  sents, prob = agent.get_attribute(filename)
  if prob then
    --print(prob)
    --print(label_list[n])
    local attribute_names = string.split(sents[1], ' ')
    local line
    for i=1,#attribute_names do
      print(attribute_names[i])
      print(prob[i])
      if attribute_name == attribute_names[i] then
        line = string.format(
          '%d %f %d\n', n, prob[i], label_list[n])
      end
    end
    if line == nil then
      line = string.format(
        '%d 0.0 %d\n', n, label_list[n])
    end
    output_fp:write(line)
  end
end

output_fp:close()

io.flush(print('Done'))

