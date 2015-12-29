
require 'nn'
require 'cunn'
require 'cudnn'

-- Delving Deep into Rectifiers: 
-- Suupassing Human-Level Performance on ImageNet Classification, 2014
-- A sufficient condition is:
-- \frac{1}{2} n_l \text{Var}[w_l] = 1, \forall l. (1)
-- where n_l = k^2 * c
-- (1) leads to a zero-mean Gaussian distribution whos std. is \sqrt{2/n_l}
-- bias is zeros
function MSRinit(net)
  local function init(module_type)
    for k, v in pairs(net:findModules(module_type)) do
      -- n = k^2 * c
      local n = v.kW * v.kH * v.nOutputPlane
      -- \sqrt{2/n}
      v.weight:normal(0,math.sqrt(2/n))
      v.bias:zero()
    end
  end
  init'nn.SpatialConvolution'
  init'nn.SpatialConvolutionMM'
  init'cudnn.SpatialConvolution'
end

