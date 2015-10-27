
-- clear the intermediate states in the model before saving to disk
local function sanitize(net)
  local list = net:listModules()
  for _,val in ipairs(list) do
    for name,field in pairs(val) do
      if torch.type(field) == 'table' then break end
      if torch.type(field) == 'cdata' then val[name] = nil end
      if name == 'homeGradBuffers' then val[name] = nil end
      if name == 'input_gpu' then val['input_gpu'] = {} end
      if name == 'gradOutput_gpu' then val['gradOutput_gpu'] = {} end
      if name == 'gradInput_gpu' then val['gradInput_gpu'] = {} end
      if (name == 'output' or name == 'gradInput') then
        val[name] = field.new()
      end
    end
  end
end

-- global
-- model
-- optimState
function conditional_save()
  sanitize(model)
  local model_file = paths.concat(
    opt.save, 'model/' .. opt.snapshot_prefix .. '_model_' .. epoch .. '.t7')
  local optimState_file = paths.concat(
    opt.save, 'model/' .. opt.snapshot_prefix .. '_optimState_' .. epoch .. '.t7')
  torch.save(model_file, model)
  torch.save(optimState_file, optimState)
  print('snapshot ' .. model_file)
  print('optimState ' .. model_file)
end

