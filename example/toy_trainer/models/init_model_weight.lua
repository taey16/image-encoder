
-- initialization from MSR
function MSRinit(net)
  local function init(name)
    for k, v in pairs(net:findModules(name)) do
      local n = v.kW*v.kH*v.nOutputPlane
      v.weight:normal(0,math.sqrt(2/n))
      v.bias:zero()
    end
  end
  init'nn.SpatialConvolution'
  if opt.backend == 'cudnn' then
    init'cudnn.SpatialConvolution'
  end
end

