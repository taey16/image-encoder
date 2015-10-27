
-- clear the intermediate states in the model before saving to disk
-- this saves lots of disk space
function sanitize(net)
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

function conditional_save(model, optimState, epoch)
  sanitize(model)
  local dump_model_path = paths.concat(opt.save, 'model_' .. epoch .. '.t7')
  local dump_optimState_path = paths.concat(opt.save, 'optimState_' .. epoch .. '.t7')
  torch.save(dump_model_path, model)
  torch.save(dump_optimState_path, optimState)
  print('Dump ' .. dump_model_path)
  print('Dump ' .. dump_optimState_path)
end

