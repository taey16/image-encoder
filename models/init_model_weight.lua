
require 'nn'
require 'cunn'
require 'cudnn'

local nn_init = {}

-- Delving Deep into Rectifiers: 
-- Suupassing Human-Level Performance on ImageNet Classification, 2014
-- A sufficient condition is:
-- \frac{1}{2} n_l \text{Var}[w_l] = 1, \forall l. (1)
-- where n_l = k^2 * c
-- (1) leads to a zero-mean Gaussian distribution whos std. is \sqrt{2/n_l}
-- bias is zeros
function nn_init.MSRinit(net)
  local function Convinit(module_type)
    for k, v in pairs(net:findModules(module_type)) do
      -- n = k^2 * c
      local n = v.kW * v.kH * v.nOutputPlane
      -- \sqrt{2/n}
      v.weight:normal(0,math.sqrt(2/n))
      v.bias:zero()

      -- As FAIR did (fb.ResNet.torch)
      -- All of the cudnn.SpatialConv. layers are followered by 
      -- cudnn.SpaatialBatchNorm.
      -- which means we did not need to set a bias vector in conv. layers
      --if cudnn.version >= 4000 then
      --  v.bias = nil
      --  v.gradBias = nil
      --else
      --  v.bias:zero()
      --end
      print(string.format(
        'Convinit Kaiming (is bias nil? %s)', tostring(v.bias == nil)))
    end
  end
  local function BNinit(module_type)
    for k,v in pairs(net:findModules(module_type)) do
      print(string.format(
        'BNinit: weight %f (is bias nil? %s)', opt.init_gamma, tostring(v.bias == nil)))
      v.weight:fill(opt.init_gamma)
      v.bias:zero()
    end
    io.flush()
  end
  local function Linearinit(module_type)
    for k,v in pairs(net:findModules(module_type)) do
      print(string.format(
        'Linearinit: bias to 0.0 (is bias nil? %s)', tostring(v.bias == nil)))
      if v.bias then
        v.bias:zero()
      end
    end
    io.flush()
  end
  local function Addinit(module_type)
    for k,v in pairs(net:findModules(module_type)) do
      print('Addinit: bias to 0.0')
      v.bias:zero()
    end
    io.flush()
  end
  Convinit'nn.SpatialConvolution'
  Convinit'nn.SpatialConvolutionMM'
  Convinit'cudnn.SpatialConvolution'
  BNinit'cudnn.SpatialBatchNormalization'
  BNinit'cudnn.BatchNormalization'
  BNinit'nn.SpatialBatchNormalization'
  BNinit'nn.BatchNormalization'
  Linearinit'nn.Linear'
  Addinit'nn.Add'
end

return nn_init

