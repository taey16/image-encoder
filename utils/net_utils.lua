
-- clear the intermediate states in the model before saving to disk
-- this saves lots of disk space
function sanitize(net)
  local _sanitize = paths.dofile('sanitize.lua')
  --[[
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
  --]]
  _sanitize.sanitize(net)
end


function print_net(net)
  local list = net:listModules()
  for _,val in ipairs(list) do
    print(val.__typename)
    for name,field in pairs(val) do
      print('--name: '.. name .. ' field: '.. torch.type(field))
    end
  end
end


function save_net(net, filename)
  sanitize(net)
  torch.save(filename, net)
  print('Save done in: ' .. filename)
end


function conditional_save(model, optimState, epoch)
  sanitize(model)
  local dump_model_path = 
    paths.concat(opt.save, 'model_' .. epoch .. '.t7')
  local dump_optimState_path = 
    paths.concat(opt.save, 'optimState_' .. epoch .. '.t7')
  torch.save(dump_model_path, model)
  torch.save(dump_optimState_path, optimState)
  print('Dump ' .. dump_model_path)
  print('Dump ' .. dump_optimState_path)
end


function paramsForEpoch(regimes, epoch)
  for _, row in ipairs(regimes) do
    if epoch >= row[1] and epoch <= row[2] then
      return { learningRate=row[3], weightDecay=row[4] }, epoch == row[1]
    end
  end
end


function paramsForIter(regimes, epoch, iter)
  local cnn_learning_rate
  if iter > opt.learning_rate_decay_start and opt.learning_rate_decay_start >= 0 then
    local frac = (iter - opt.learning_rate_decay_start) / opt.learning_rate_decay_every
    local decay_factor = math.pow(0.5, frac)
    learning_rate = learning_rate * decay_factor
    cnn_learning_rate = cnn_learning_rate * decay_factor
  end
  return {learningRate = cnn_learning_rate}
end


function reset_optimState(params)
  -- options for optim.std
  local optimState = {
    -- laarning rate
    learningRate = params.learningRate,
    -- vector of individual learning rates
    --learningRates= params.learningRates,
    -- learning rate decay
    learningRateDecay = 0.0,
    -- weight decay
    weightDecay = params.weightDecay,
    -- vector of individual weight decays
    --weightDecays= params.weightDecays
    -- momentum
    momentum = opt.momentum,
    -- dampening for momentum
    dampening = 0.0,
  }
  return optimState
end

function conv2prototxt(idx, module_obj, layer_name, bottom, top)
  local str_ = 
    ("layer{\n  name: \"%s\"\n  type: \"Convolution\"\n  bottom: \"%s\"\n  top: \"%s\"\n  convolution_param {\n    num_output: %d pad_h: %d pad_w: %d kernel_h: %d kernel_w: %d stride_h: %d stride_w: %d engin: CUDNN\n  }\n}"):format(
      layer_name, bottom, top,
      module_obj.nOutputPlane,
      module_obj.padH,
      module_obj.padW,
      module_obj.kH,
      module_obj.kW,
      module_obj.dH,
      module_obj.dW)
  print(str_)
end


function relu2prototxt(idx, module_obj, layer_name, bottom, top)
  local str_ = 
    ("layer{ name: \"%s\" type: \"ReLU\" bottom: \"%s\" top: \"%s\" engin: CUDNN}"):format(
      layer_name, bottom, top)
  print(str_)
end

function pool2prototxt(idx, module_obj, layer_name, bottom, top, pool_type)
  local str_ = 
    ("layer{\n  name: \"%s\"\n  type: \"Pooling\"\n  bottom: \"%s\"\n  top: \"%s\"\n  pooling_param {\n    pool: %s  pad_h: %d pad_w: %d kernel_h: %d kernel_w: %d stride_h: %d stride_w: %d engin: CUDNN\n  }\n}"):format(
      layer_name, bottom, top, pool_type,
      module_obj.padH,
      module_obj.padW,
      module_obj.kH,
      module_obj.kW,
      module_obj.dH,
      module_obj.dW)
  print(str_)
end


function linear2prototxt(idx, module_obj, layer_name, bottom, top)
  local str_ = 
    ("layer{\n  name: \"%s\"\n  type: \"InnerProduct\"\n  bottom: \"%s\"\n  top: \"%s\"\n  inner_product_param {\n    num_output: %d\n  }\n}"):format(
      layer_name, bottom, top,
      module_obj.weight:size(2))
  print(str_)
end


function softmax2prototxt(idx, module_obj, layer_name, bottom, top)
  local str_ = 
    ("layer{\n  name: \"%s\"\n  type: \"Softmax\"\n  bottom: \"%s\"\n  top: \"%s\"\n  engin: CUDNN\n}"):format(
      layer_name, bottom, top)
  print(str_)
end


function generate_Caffe_prototxt(net)
  local node_id = 1
  while (node_id <= #net.modules) do 
    --if node_id > 11 then break end
    if net.modules[node_id].__typename == 'nn.Sequential' then
      generate_Caffe_prototxt(net.modules[node_id])
    elseif net.modules[node_id].__typename == 'nn.DepthConcat' then
      print('nn.DepthConcat')
      generate_Caffe_prototxt(net.modules[node_id])
    elseif net.modules[node_id].__typename == 'cudnn.SpatialConvolution' then
      local module_name = ("conv%d"):format(node_id)
      conv2prototxt(node_id, net.modules[node_id], module_name, prev_name, module_name)
      prev_name = module_name
    elseif net.modules[node_id].__typename == 'cudnn.ReLU' then
      local module_name = ("relu%d"):format(node_id)
      if net.modules[node_id].inplace then
        relu2prototxt(node_id, net.modules[node_id], module_name, prev_name, module_name)
      else
        relu2prototxt(node_id, net.modules[node_id],  module_name, prev_name, module_name)
      end
      prev_name = module_name
    elseif net.modules[node_id].__typename == 'cudnn.SpatialMaxPooling' then
      local module_name = ("maxpool%d"):format(node_id)
      pool2prototxt(node_id, net.modules[node_id], module_name, prev_name, module_name, "MAX")
      prev_name = module_name
    elseif net.modules[node_id].__typename == 'cudnn.SpatialAveragePooling' then
      local module_name = ("maxpool%d"):format(node_id)
      pool2prototxt(node_id, net.modules[node_id], module_name, prev_name, module_name, "AVE")
      prev_name = module_name
    elseif net.modules[node_id].__typename == 'nn.Linear' then
      local module_name = ("linear%d"):format(node_id)
      linear2prototxt(node_id, net.modules[node_id], module_name, prev_name, module_name)
      prev_name = module_name
    elseif net.modules[node_id].__typename == 'cudnn.LogSoftMax' or 
           net.modules[node_id].__typename == 'cudnn.SoftMax' then
      local module_name = ("softmax%d"):format(node_id)
      softmax2prototxt(node_id, net.modules[node_id], module_name, prev_name, module_name)
      prev_name = module_name
    end
    node_id = node_id + 1
  end
end
