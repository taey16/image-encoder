require 'nn'
require 'cunn'
require 'cudnn'
require 'torch'


local function absorb_bn(w, b, mean, std, affine, gamma, beta)
  w:cmul(std:view(w:size(1),1):repeatTensor(1,w:size(2)))
  b:add(-mean):cmul(std)

  if affine then
    w:cmul(gamma:view(w:size(1),1):repeatTensor(1,w:size(2)))
    b:cmul(gamma):add(beta)
  end
end


local function BN_absorber(x)
  local i = 1
  while (i <= #x.modules) do
    if x.modules[i].__typename == 'nn.Sequential' then
      print(string.format('(%d) nn.Sequential', i))
      BN_absorber(x.modules[i])
    elseif x.modules[i].__typename == 'nn.Parallel' then
      BN_absorber(x.modules[i])
    elseif x.modules[i].__typename == 'nn.Concat' then
      BN_absorber(x.modules[i])
    elseif x.modules[i].__typename == 'nn.DataParallel' then
      BN_absorber(x.modules[i])
    elseif x.modules[i].__typename == 'nn.ModelParallel' then
      BN_absorber(x.modules[i])
    elseif x.modules[i].__typename == 'nn.DepthConcat' then
      print(string.format('(%d) nn.DepthConcat', i))
      BN_absorber(x.modules[i])
    elseif x.modules[i].__typename == 'nn.ConcatTable' then
      print(string.format('(%d) nn.ConcatTable', i))
      BN_absorber(x.modules[i])
    else
      assert(x.modules[i].__typename ~= 'cudnn.BatchNormalization', 
        'cudnn.torch R4 doesnot support per-activation BN')
      if x.modules[i].__typename == 'nn.SpatialBatchNormalization' or
         x.modules[i].__typename == 'nn.BatchNormalization' or 
         x.modules[i].__typename == 'cudnn.SpatialBatchNormalization' then
        if x.modules[i-1] and
          (x.modules[i-1].__typename == 'nn.Linear' or
           x.modules[i-1].__typename == 'nn.SpatialConvolution' or
           x.modules[i-1].__typename == 'cudnn.SpatialConvolution' or
           x.modules[i-1].__typename == 'nn.SpatialConvolutionMM') then

          if x.modules[i-1].gradWeight then
            x.modules[i-1].gradWeight = nil
          end
          if x.modules[i-1].gradBias then
            x.mdoules[i-1].gradBias = nil
          end

           -- force weight to be in 2-dim
          local weight = x.modules[i-1].weight
          weight = weight:view(weight:size(1), weight:nElement()/weight:size(1))

          -- in ConvInit:init_model_weight.lua
          -- all bias term in cudnn.SpatialConvolution are removed so that
          -- we need to initialize it with a zero vector
          if cudnn.version >= 4000 and x.modules[i-1].bias == nil then
            x.modules[i-1].bias = torch.CudaTensor(x.modules[i-1].nOutputPlane):zero()
          end

          if x.modules[i].__typename == 'nn.SpatialBatchNormalization' or
             x.modules[i].__typename == 'nn.BatchNormalization' then
             -- (TODO) does not work. I dont know
            dmy = 1
          else
            -- remove BN
            absorb_bn(weight,
              x.modules[i-1].bias,
              x.modules[i].running_mean,
              x.modules[i].running_std,
              x.modules[i].affine,
              x.modules[i].weight,
              x.modules[i].bias)
            x:remove(i)
            i = i - 1
          end
        else
          assert(false, 
            'Convolution module must exist right before batch normalization layer')
        end
      end
    end
    i = i + 1
  end
  collectgarbage()
  return x
end

return BN_absorber

