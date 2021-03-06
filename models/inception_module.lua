
require 'cudnn'
require 'cunn'


--  FAST AND ACCURATE DEEP NETWORK LEARNING BY EXPONENTIAL LINEAR UNITS (ELUS), ICLR, 2016
function resception_elu_module(mixed_id, input_size, config)
  local std_epsilon = 0.0010000000475
  local layer_id = mixed_id
  local mixed = nn.DepthConcat(2)

  if layer_id == 1 or layer_id == 2 or layer_id == 3 then
    -- similar to fig.4,5
    -- 1x1
    -- 1x1 -> 5x5
    -- 1x1 -> 3x3 -> 3x3
    -- 3x3(1,same,avg-pool) -> 1x1
    local mixed_conv = nn.Sequential()
      -- valid
    mixed_conv:add(cudnn.SpatialConvolution(input_size, config[1][1], 1, 1, 1, 1, 0, 0))
    mixed_conv:add(nn.ELU(1.0, true))

    local shortcut_mixed_tower_conv = nn.Sequential()
      shortcut_mixed_tower_conv:add(cudnn.SpatialConvolution(input_size, config[2][2], 1, 1, 1, 1, 0, 0))

    local mixed_tower_conv = nn.Sequential()
      mixed_tower_conv:add(cudnn.SpatialConvolution(input_size, config[2][1], 1, 1, 1, 1, 0, 0))
      mixed_tower_conv:add(nn.ELU(1.0, true))
      mixed_tower_conv:add(cudnn.SpatialConvolution(config[2][1], config[2][2], 5, 5, 1, 1, 2, 2))
    local concat_mixed_tower_conv = nn.ConcatTable()
      concat_mixed_tower_conv:add(mixed_tower_conv)
      concat_mixed_tower_conv:add(shortcut_mixed_tower_conv)
    local add_concat_mixed_tower_conv = nn.Sequential()
      add_concat_mixed_tower_conv:add(concat_mixed_tower_conv)
      add_concat_mixed_tower_conv:add(nn.CAddTable())
      add_concat_mixed_tower_conv:add(nn.ELU(1.0, true))

    local shortcut_mixed_tower_1_conv = nn.Sequential()
      shortcut_mixed_tower_1_conv:add(cudnn.SpatialConvolution(input_size, config[3][3], 1, 1, 1, 1, 0, 0))

    local mixed_tower_1_conv = nn.Sequential()
      mixed_tower_1_conv:add(cudnn.SpatialConvolution(input_size, config[3][1], 1, 1, 1, 1, 0, 0))
      mixed_tower_1_conv:add(nn.ELU(1.0, true))
      mixed_tower_1_conv:add(cudnn.SpatialConvolution(config[3][1], config[3][2], 3, 3, 1, 1, 1, 1))
      mixed_tower_1_conv:add(nn.ELU(1.0, true))
      mixed_tower_1_conv:add(cudnn.SpatialConvolution(config[3][2], config[3][3], 3, 3, 1, 1, 1, 1))

    local concat_mixed_tower_1_conv = nn.ConcatTable()
      concat_mixed_tower_1_conv:add(mixed_tower_1_conv)
      concat_mixed_tower_1_conv:add(shortcut_mixed_tower_1_conv)
    local add_concat_mixed_tower_1_conv = nn.Sequential()
      add_concat_mixed_tower_1_conv:add(concat_mixed_tower_1_conv)
      add_concat_mixed_tower_1_conv:add(nn.CAddTable())
      add_concat_mixed_tower_1_conv:add(nn.ELU(1.0, true))

    local mixed_tower_2_conv_pool = nn.Sequential()
      mixed_tower_2_conv_pool:add(cudnn.SpatialAveragePooling(3, 3, 1, 1, 1, 1))
      mixed_tower_2_conv_pool:add(cudnn.SpatialConvolution(input_size, config[4][1], 1, 1, 1, 1, 0, 0))
      mixed_tower_2_conv_pool:add(nn.ELU(1.0, true))
      mixed:add(mixed_conv)
      mixed:add(add_concat_mixed_tower_conv)
      mixed:add(add_concat_mixed_tower_1_conv)
      mixed:add(mixed_tower_2_conv_pool)
  elseif layer_id == 4 then
    -- similar to fig.10
    -- 3x3(2,valid)
    -- 1x1 -> 3x3(1,same) -> 3x3(2,valid)
    -- 3x3(2,valid,max-pool)
    local mixed_conv = nn.Sequential()
      mixed_conv:add(cudnn.SpatialConvolution(input_size, config[1][1], 3, 3, 2, 2, 0, 0))
      mixed_conv:add(nn.ELU(1.0, true))

    local shortcut_mixed_tower_1_conv = nn.Sequential()
      shortcut_mixed_tower_1_conv:add(cudnn.SpatialConvolution(input_size, config[2][3], 3, 3, 2, 2, 0, 0))

    local mixed_tower_1_conv = nn.Sequential()
      mixed_tower_1_conv:add(cudnn.SpatialConvolution(input_size, config[2][1], 1, 1, 1, 1, 0, 0))
      mixed_tower_1_conv:add(nn.ELU(1.0, true))
      mixed_tower_1_conv:add(cudnn.SpatialConvolution(config[2][1], config[2][2], 3, 3, 1, 1, 1, 1))
      mixed_tower_1_conv:add(nn.ELU(1.0, true))
      mixed_tower_1_conv:add(cudnn.SpatialConvolution(config[2][2], config[2][3], 3, 3, 2, 2, 0, 0))

    local concat_mixed_tower_1_conv = nn.ConcatTable()
      concat_mixed_tower_1_conv:add(mixed_tower_1_conv)
      concat_mixed_tower_1_conv:add(shortcut_mixed_tower_1_conv)
    local add_concat_mixed_tower_1_conv = nn.Sequential()
      add_concat_mixed_tower_1_conv:add(concat_mixed_tower_1_conv)
      add_concat_mixed_tower_1_conv:add(nn.CAddTable())
      add_concat_mixed_tower_1_conv:add(nn.ELU(1.0, true))

    local mixed_pool = nn.Sequential()
      mixed_pool:add(cudnn.SpatialMaxPooling(3, 3, 2, 2, 0, 0))
      mixed:add(mixed_conv)
      mixed:add(add_concat_mixed_tower_1_conv)
      mixed:add(mixed_pool)
  elseif layer_id == 5 or layer_id == 6 or layer_id == 7 or layer_id == 8 then
    -- figure 6
    -- 1x1
    -- 1x1 -> 7x1 -> 1x7
    -- 1x1 -> 1x7 -> 7x1 -> 1x7 -> 7x1
    -- 3x3(1,same,avg=pool) -> 1x1
    local mixed_conv = nn.Sequential()
      mixed_conv:add(cudnn.SpatialConvolution(input_size, config[1][1], 1, 1, 1, 1, 0, 0))
      mixed_conv:add(nn.ELU(1.0, true))

    local shortcut_mixed_tower_conv = nn.Sequential()
      shortcut_mixed_tower_conv:add(cudnn.SpatialConvolution(input_size, config[2][3], 1, 1, 1, 1, 0, 0))

    local mixed_tower_conv = nn.Sequential()
      mixed_tower_conv:add(cudnn.SpatialConvolution(input_size, config[2][1], 1, 1, 1, 1, 0, 0))
      mixed_tower_conv:add(nn.ELU(1.0, true))
      mixed_tower_conv:add(cudnn.SpatialConvolution(config[2][1], config[2][2], 7, 1, 1, 1, 3, 0))
      mixed_tower_conv:add(nn.ELU(1.0, true))
      mixed_tower_conv:add(cudnn.SpatialConvolution(config[2][2], config[2][3], 1, 7, 1, 1, 0, 3))

    local concat_mixed_tower_conv = nn.ConcatTable()
      concat_mixed_tower_conv:add(mixed_tower_conv)
      concat_mixed_tower_conv:add(shortcut_mixed_tower_conv)
    local add_concat_mixed_tower_conv = nn.Sequential()
      add_concat_mixed_tower_conv:add(concat_mixed_tower_conv)
      add_concat_mixed_tower_conv:add(nn.CAddTable())
      add_concat_mixed_tower_conv:add(nn.ELU(1.0, true))

    local shortcut_mixed_tower_1_conv = nn.Sequential()
      shortcut_mixed_tower_1_conv:add(cudnn.SpatialConvolution(input_size, config[3][5], 1, 1, 1, 1, 0, 0))

    local mixed_tower_1_conv = nn.Sequential()
      mixed_tower_1_conv:add(cudnn.SpatialConvolution(input_size, config[3][1], 1, 1, 1, 1, 0, 0))
      mixed_tower_1_conv:add(nn.ELU(1.0, true))
      mixed_tower_1_conv:add(cudnn.SpatialConvolution(config[3][1], config[3][2], 1, 7, 1, 1, 0, 3))
      mixed_tower_1_conv:add(nn.ELU(1.0, true))
      mixed_tower_1_conv:add(cudnn.SpatialConvolution(config[3][2], config[3][3], 7, 1, 1, 1, 3, 0))
      mixed_tower_1_conv:add(nn.ELU(1.0, true))
      mixed_tower_1_conv:add(cudnn.SpatialConvolution(config[3][3], config[3][4], 1, 7, 1, 1, 0, 3))
      mixed_tower_1_conv:add(nn.ELU(1.0, true))
      mixed_tower_1_conv:add(cudnn.SpatialConvolution(config[3][4], config[3][5], 7, 1, 1, 1, 3, 0))
    local concat_mixed_tower_1_conv = nn.ConcatTable()
      concat_mixed_tower_1_conv:add(mixed_tower_1_conv)
      concat_mixed_tower_1_conv:add(shortcut_mixed_tower_1_conv)
    local add_concat_mixed_tower_1_conv = nn.Sequential()
      add_concat_mixed_tower_1_conv:add(concat_mixed_tower_1_conv)
      add_concat_mixed_tower_1_conv:add(nn.CAddTable())
      add_concat_mixed_tower_1_conv:add(nn.ELU(1.0, true))

    local mixed_tower_2_conv_pool = nn.Sequential()
      mixed_tower_2_conv_pool:add(cudnn.SpatialAveragePooling(3, 3, 1, 1, 1, 1))
      mixed_tower_2_conv_pool:add(cudnn.SpatialConvolution(input_size, config[4][1], 1, 1, 1, 1, 0, 0))
      mixed_tower_2_conv_pool:add(nn.ELU(1.0, true))
      mixed:add(mixed_conv)
      mixed:add(add_concat_mixed_tower_conv)
      mixed:add(add_concat_mixed_tower_1_conv)
      mixed:add(mixed_tower_2_conv_pool)
  elseif layer_id == 9 then
    -- similar to fig.10
    -- 1x1 -> 3x3(2,valid)
    -- 1x1 -> 7x1 -> 1x7 -> 3x3(2,valid)
    -- 3x3(2,valid,max-pool)
    local shortcut_mixed_conv = nn.Sequential()
      shortcut_mixed_conv:add(cudnn.SpatialConvolution(input_size, config[1][2], 3, 3, 2, 2, 0, 0))

    local mixed_conv = nn.Sequential()
      mixed_conv:add(cudnn.SpatialConvolution(input_size, config[1][1], 1, 1, 1, 1, 0, 0))
      mixed_conv:add(nn.ELU(1.0, true))
      mixed_conv:add(cudnn.SpatialConvolution(config[1][1], config[1][2], 3, 3, 2, 2, 0, 0))

    local concat_mixed_conv = nn.ConcatTable()
      concat_mixed_conv:add(mixed_conv)
      concat_mixed_conv:add(shortcut_mixed_conv)
    local add_concat_mixed_conv = nn.Sequential()
      add_concat_mixed_conv:add(concat_mixed_conv)
      add_concat_mixed_conv:add(nn.CAddTable())
      add_concat_mixed_conv:add(nn.ELU(1.0, true))

    local shortcut_mixed_tower_conv = nn.Sequential()
    shortcut_mixed_tower_conv:add(cudnn.SpatialConvolution(input_size, config[2][4], 3, 3, 2, 2, 0, 0))

    local mixed_tower_conv = nn.Sequential()
      mixed_tower_conv:add(cudnn.SpatialConvolution(input_size, config[2][1], 1, 1, 1, 1, 0, 0))
      mixed_tower_conv:add(nn.ELU(1.0, true))
      mixed_tower_conv:add(cudnn.SpatialConvolution(config[2][1], config[2][2], 7, 1, 1, 1, 3, 0))
      mixed_tower_conv:add(nn.ELU(1.0, true))
      mixed_tower_conv:add(cudnn.SpatialConvolution(config[2][2], config[2][3], 1, 7, 1, 1, 0, 3))
      mixed_tower_conv:add(nn.ELU(1.0, true))
      mixed_tower_conv:add(cudnn.SpatialConvolution(config[2][3], config[2][4], 3, 3, 2, 2, 0, 0))
    local concat_mixed_tower_conv = nn.ConcatTable()
      concat_mixed_tower_conv:add(mixed_tower_conv)
      concat_mixed_tower_conv:add(shortcut_mixed_tower_conv)
    local add_concat_mixed_tower_conv = nn.Sequential()
      add_concat_mixed_tower_conv:add(concat_mixed_tower_conv)
      add_concat_mixed_tower_conv:add(nn.CAddTable())
      add_concat_mixed_tower_conv:add(nn.ELU(1.0, true))

    local mixed_pool = nn.Sequential()
      mixed_pool:add(cudnn.SpatialMaxPooling(3, 3, 2, 2, 0, 0))
      mixed:add(add_concat_mixed_conv)
      mixed:add(add_concat_mixed_tower_conv)
      mixed:add(mixed_pool)
  elseif layer_id == 10 or layer_id == 11 then
    -- fig. 7
    -- 1x1
    -- 1x1 -> 3x1
    --     -> 1x3 
    -- 1x1 -> 3x3(1,same) -> 3x1
    --                    -> 1x3
    -- 3x3(1,same,avg-pool) -> 1x1
    local mixed_conv = nn.Sequential()
      mixed_conv:add(cudnn.SpatialConvolution(input_size, config[1][1], 1, 1, 1, 1, 0, 0))
      mixed_conv:add(nn.ELU(1.0, true))

    local shortcut_mixed_tower_conv = nn.Sequential()
      shortcut_mixed_tower_conv:add(cudnn.SpatialConvolution(input_size, config[2][2]+config[2][3], 1, 1, 1, 1, 0, 0))

    local mixed_tower_conv = nn.Sequential()
      mixed_tower_conv:add(cudnn.SpatialConvolution(input_size, config[2][1], 1, 1, 1, 1, 0, 0))
      mixed_tower_conv:add(nn.ELU(1.0, true))
      local mixed_tower_expand = nn.Sequential()
        local mixed_expand = nn.DepthConcat(2)
        local mixed_expand_1 = nn.Sequential()
          mixed_expand_1:add(cudnn.SpatialConvolution(config[2][1], config[2][2], 3, 1, 1, 1, 1, 0))
        local mixed_expand_2 = nn.Sequential()
          mixed_expand_2:add(cudnn.SpatialConvolution(config[2][1], config[2][3], 1, 3, 1, 1, 0, 1))
        mixed_expand:add(mixed_expand_1)
        mixed_expand:add(mixed_expand_2)
        mixed_tower_expand:add(mixed_expand)
        mixed_tower_conv:add(mixed_tower_expand)
 
    local concat_mixed_tower_conv = nn.ConcatTable()
      concat_mixed_tower_conv:add(mixed_tower_conv)
      concat_mixed_tower_conv:add(shortcut_mixed_tower_conv)

    local add_concat_mixed_tower_conv = nn.Sequential()
    add_concat_mixed_tower_conv:add(concat_mixed_tower_conv)
    add_concat_mixed_tower_conv:add(nn.CAddTable())
    add_concat_mixed_tower_conv:add(nn.ELU(1.0, true))

    local shortcut_mixed_tower_1_conv = nn.Sequential()
      shortcut_mixed_tower_1_conv:add(cudnn.SpatialConvolution(input_size, config[3][3]+config[3][4], 1, 1, 1, 1, 0, 0))

    local mixed_tower_1_conv = nn.Sequential()
      mixed_tower_1_conv:add(cudnn.SpatialConvolution(input_size, config[3][1], 1, 1, 1, 1, 0, 0))
      mixed_tower_1_conv:add(nn.ELU(1.0, true))
      mixed_tower_1_conv:add(cudnn.SpatialConvolution(config[3][1], config[3][2], 3, 3, 1, 1, 1, 1))
      mixed_tower_1_conv:add(nn.ELU(1.0, true))
      local mixed_tower_1_expand = nn.Sequential()
      local mixed_1_expand = nn.DepthConcat(2)
      local mixed_1_expand_1 = nn.Sequential()
        mixed_1_expand_1:add(cudnn.SpatialConvolution(config[3][2], config[3][3], 3, 1, 1, 1, 1, 0))
      local mixed_1_expand_2 = nn.Sequential()
        mixed_1_expand_2:add(cudnn.SpatialConvolution(config[3][2], config[3][4], 1, 3, 1, 1, 0, 1))
        mixed_1_expand:add(mixed_1_expand_1)
        mixed_1_expand:add(mixed_1_expand_2)
      mixed_tower_1_expand:add(mixed_1_expand)
      mixed_tower_1_conv:add(mixed_tower_1_expand)

    local concat_mixed_tower_1_conv = nn.ConcatTable()
      concat_mixed_tower_1_conv:add(mixed_tower_1_conv)
      concat_mixed_tower_1_conv:add(shortcut_mixed_tower_1_conv)
    local add_concat_mixed_tower_1_conv = nn.Sequential()
      add_concat_mixed_tower_1_conv:add(concat_mixed_tower_1_conv)
      add_concat_mixed_tower_1_conv:add(nn.CAddTable())
      add_concat_mixed_tower_1_conv:add(nn.ELU(1.0, true))

    local mixed_tower_2_pool = nn.Sequential()
      mixed_tower_2_pool:add(cudnn.SpatialAveragePooling(3, 3, 1, 1, 1, 1))
      mixed_tower_2_pool:add(cudnn.SpatialConvolution(input_size, config[4][1], 1, 1, 1, 1, 0, 0))
      mixed_tower_2_pool:add(nn.ELU(1.0, true))
      mixed:add(mixed_conv)
      mixed:add(add_concat_mixed_tower_conv)
      mixed:add(add_concat_mixed_tower_1_conv)
      mixed:add(mixed_tower_2_pool)
  end
  return mixed
end


function resception_module(mixed_id, input_size, config)
  local std_epsilon = 0.0010000000475
  local layer_id = mixed_id
  local mixed = nn.DepthConcat(2)

  if layer_id == 1 or layer_id == 2 or layer_id == 3 then
    -- similar to fig.4,5
    -- 1x1
    -- 1x1 -> 5x5
    -- 1x1 -> 3x3 -> 3x3
    -- 3x3(1,same,avg-pool) -> 1x1
    local mixed_conv = nn.Sequential()
      -- valid
    mixed_conv:add(cudnn.SpatialConvolution(input_size, config[1][1], 1, 1, 1, 1, 0, 0))
    mixed_conv:add(cudnn.SpatialBatchNormalization(config[1][1], std_epsilon, nil, true))
    mixed_conv:add(cudnn.ReLU(true))

    local shortcut_mixed_tower_conv = nn.Sequential()
      shortcut_mixed_tower_conv:add(cudnn.SpatialConvolution(input_size, config[2][2], 1, 1, 1, 1, 0, 0))
      shortcut_mixed_tower_conv:add(cudnn.SpatialBatchNormalization(config[2][2], std_epsilon, nil, true))

    local mixed_tower_conv = nn.Sequential()
      mixed_tower_conv:add(cudnn.SpatialConvolution(input_size, config[2][1], 1, 1, 1, 1, 0, 0))
      mixed_tower_conv:add(cudnn.SpatialBatchNormalization(config[2][1], std_epsilon, nil, true))
      mixed_tower_conv:add(cudnn.ReLU(true))
      mixed_tower_conv:add(cudnn.SpatialConvolution(config[2][1], config[2][2], 5, 5, 1, 1, 2, 2))
      mixed_tower_conv:add(cudnn.SpatialBatchNormalization(config[2][2], std_epsilon, nil, true))
      --mixed_tower_conv:add(cudnn.ReLU(true))
    local concat_mixed_tower_conv = nn.ConcatTable()
      concat_mixed_tower_conv:add(mixed_tower_conv)
      concat_mixed_tower_conv:add(shortcut_mixed_tower_conv)
    local add_concat_mixed_tower_conv = nn.Sequential()
      add_concat_mixed_tower_conv:add(concat_mixed_tower_conv)
      add_concat_mixed_tower_conv:add(nn.CAddTable())
      add_concat_mixed_tower_conv:add(cudnn.ReLU(true))

    local shortcut_mixed_tower_1_conv = nn.Sequential()
      shortcut_mixed_tower_1_conv:add(cudnn.SpatialConvolution(input_size, config[3][3], 1, 1, 1, 1, 0, 0))
      shortcut_mixed_tower_1_conv:add(cudnn.SpatialBatchNormalization(config[3][3], std_epsilon, nil, true))

    local mixed_tower_1_conv = nn.Sequential()
      mixed_tower_1_conv:add(cudnn.SpatialConvolution(input_size, config[3][1], 1, 1, 1, 1, 0, 0))
      mixed_tower_1_conv:add(cudnn.SpatialBatchNormalization(config[3][1], std_epsilon, nil, true))
      mixed_tower_1_conv:add(cudnn.ReLU(true))
      mixed_tower_1_conv:add(cudnn.SpatialConvolution(config[3][1], config[3][2], 3, 3, 1, 1, 1, 1))
      mixed_tower_1_conv:add(cudnn.SpatialBatchNormalization(config[3][2], std_epsilon, nil, true))
      mixed_tower_1_conv:add(cudnn.ReLU(true))
      mixed_tower_1_conv:add(cudnn.SpatialConvolution(config[3][2], config[3][3], 3, 3, 1, 1, 1, 1))
      mixed_tower_1_conv:add(cudnn.SpatialBatchNormalization(config[3][3], std_epsilon, nil, true))
      --mixed_tower_1_conv:add(cudnn.ReLU(true))

    local concat_mixed_tower_1_conv = nn.ConcatTable()
      concat_mixed_tower_1_conv:add(mixed_tower_1_conv)
      concat_mixed_tower_1_conv:add(shortcut_mixed_tower_1_conv)
    local add_concat_mixed_tower_1_conv = nn.Sequential()
      add_concat_mixed_tower_1_conv:add(concat_mixed_tower_1_conv)
      add_concat_mixed_tower_1_conv:add(nn.CAddTable())
      add_concat_mixed_tower_1_conv:add(cudnn.ReLU(true))

    local mixed_tower_2_conv_pool = nn.Sequential()
      mixed_tower_2_conv_pool:add(cudnn.SpatialAveragePooling(3, 3, 1, 1, 1, 1))
      mixed_tower_2_conv_pool:add(cudnn.SpatialConvolution(input_size, config[4][1], 1, 1, 1, 1, 0, 0))
      mixed_tower_2_conv_pool:add(cudnn.SpatialBatchNormalization(config[4][1], std_epsilon, nil, true))
      mixed_tower_2_conv_pool:add(cudnn.ReLU(true))
      mixed:add(mixed_conv)
      mixed:add(add_concat_mixed_tower_conv)
      mixed:add(add_concat_mixed_tower_1_conv)
      mixed:add(mixed_tower_2_conv_pool)
  elseif layer_id == 4 then
    -- similar to fig.10
    -- 3x3(2,valid)
    -- 1x1 -> 3x3(1,same) -> 3x3(2,valid)
    -- 3x3(2,valid,max-pool)
    local mixed_conv = nn.Sequential()
      mixed_conv:add(cudnn.SpatialConvolution(input_size, config[1][1], 3, 3, 2, 2, 0, 0))
      mixed_conv:add(cudnn.SpatialBatchNormalization(config[1][1], std_epsilon, nil, true))
      mixed_conv:add(cudnn.ReLU(true))

    local shortcut_mixed_tower_1_conv = nn.Sequential()
      shortcut_mixed_tower_1_conv:add(cudnn.SpatialConvolution(input_size, config[2][3], 3, 3, 2, 2, 0, 0))
      shortcut_mixed_tower_1_conv:add(cudnn.SpatialBatchNormalization(config[2][3], std_epsilon, nil, true))

    local mixed_tower_1_conv = nn.Sequential()
      mixed_tower_1_conv:add(cudnn.SpatialConvolution(input_size, config[2][1], 1, 1, 1, 1, 0, 0))
      mixed_tower_1_conv:add(cudnn.SpatialBatchNormalization(config[2][1], std_epsilon, nil, true))
      mixed_tower_1_conv:add(cudnn.ReLU(true))
      mixed_tower_1_conv:add(cudnn.SpatialConvolution(config[2][1], config[2][2], 3, 3, 1, 1, 1, 1))
      mixed_tower_1_conv:add(cudnn.SpatialBatchNormalization(config[2][2], std_epsilon, nil, true))
      mixed_tower_1_conv:add(cudnn.ReLU(true))
      mixed_tower_1_conv:add(cudnn.SpatialConvolution(config[2][2], config[2][3], 3, 3, 2, 2, 0, 0))
      mixed_tower_1_conv:add(cudnn.SpatialBatchNormalization(config[2][3], std_epsilon, nil, true))
      --mixed_tower_1_conv:add(cudnn.ReLU(true))

    local concat_mixed_tower_1_conv = nn.ConcatTable()
      concat_mixed_tower_1_conv:add(mixed_tower_1_conv)
      concat_mixed_tower_1_conv:add(shortcut_mixed_tower_1_conv)
    local add_concat_mixed_tower_1_conv = nn.Sequential()
      add_concat_mixed_tower_1_conv:add(concat_mixed_tower_1_conv)
      add_concat_mixed_tower_1_conv:add(nn.CAddTable())
      add_concat_mixed_tower_1_conv:add(cudnn.ReLU(true))

    local mixed_pool = nn.Sequential()
      mixed_pool:add(cudnn.SpatialMaxPooling(3, 3, 2, 2, 0, 0))
      mixed:add(mixed_conv)
      mixed:add(add_concat_mixed_tower_1_conv)
      mixed:add(mixed_pool)
  elseif layer_id == 5 or layer_id == 6 or layer_id == 7 or layer_id == 8 then
    -- figure 6
    -- 1x1
    -- 1x1 -> 7x1 -> 1x7
    -- 1x1 -> 1x7 -> 7x1 -> 1x7 -> 7x1
    -- 3x3(1,same,avg=pool) -> 1x1
    local mixed_conv = nn.Sequential()
      mixed_conv:add(cudnn.SpatialConvolution(input_size, config[1][1], 1, 1, 1, 1, 0, 0))
      mixed_conv:add(cudnn.SpatialBatchNormalization(config[1][1], std_epsilon, nil, true))
      mixed_conv:add(cudnn.ReLU(true))

    local shortcut_mixed_tower_conv = nn.Sequential()
      shortcut_mixed_tower_conv:add(cudnn.SpatialConvolution(input_size, config[2][3], 1, 1, 1, 1, 0, 0))
      shortcut_mixed_tower_conv:add(cudnn.SpatialBatchNormalization(config[2][3], std_epsilon, nil, true))

    local mixed_tower_conv = nn.Sequential()
      mixed_tower_conv:add(cudnn.SpatialConvolution(input_size, config[2][1], 1, 1, 1, 1, 0, 0))
      mixed_tower_conv:add(cudnn.SpatialBatchNormalization(config[2][1], std_epsilon, nil, true))
      mixed_tower_conv:add(cudnn.ReLU(true))
      mixed_tower_conv:add(cudnn.SpatialConvolution(config[2][1], config[2][2], 7, 1, 1, 1, 3, 0))
      mixed_tower_conv:add(cudnn.SpatialBatchNormalization(config[2][2], std_epsilon, nil, true))
      mixed_tower_conv:add(cudnn.ReLU(true))
      mixed_tower_conv:add(cudnn.SpatialConvolution(config[2][2], config[2][3], 1, 7, 1, 1, 0, 3))
      mixed_tower_conv:add(cudnn.SpatialBatchNormalization(config[2][3], std_epsilon, nil, true))
      --mixed_tower_conv:add(cudnn.ReLU(true))

    local concat_mixed_tower_conv = nn.ConcatTable()
      concat_mixed_tower_conv:add(mixed_tower_conv)
      concat_mixed_tower_conv:add(shortcut_mixed_tower_conv)
    local add_concat_mixed_tower_conv = nn.Sequential()
      add_concat_mixed_tower_conv:add(concat_mixed_tower_conv)
      add_concat_mixed_tower_conv:add(nn.CAddTable())
      add_concat_mixed_tower_conv:add(cudnn.ReLU(true))

    local shortcut_mixed_tower_1_conv = nn.Sequential()
      shortcut_mixed_tower_1_conv:add(cudnn.SpatialConvolution(input_size, config[3][5], 1, 1, 1, 1, 0, 0))
      shortcut_mixed_tower_1_conv:add(cudnn.SpatialBatchNormalization(config[3][5], std_epsilon, nil, true))

    local mixed_tower_1_conv = nn.Sequential()
      mixed_tower_1_conv:add(cudnn.SpatialConvolution(input_size, config[3][1], 1, 1, 1, 1, 0, 0))
      mixed_tower_1_conv:add(cudnn.SpatialBatchNormalization(config[3][1], std_epsilon, nil, true))
      mixed_tower_1_conv:add(cudnn.ReLU(true))
      mixed_tower_1_conv:add(cudnn.SpatialConvolution(config[3][1], config[3][2], 1, 7, 1, 1, 0, 3))
      mixed_tower_1_conv:add(cudnn.SpatialBatchNormalization(config[3][2], std_epsilon, nil, true))
      mixed_tower_1_conv:add(cudnn.ReLU(true))
      mixed_tower_1_conv:add(cudnn.SpatialConvolution(config[3][2], config[3][3], 7, 1, 1, 1, 3, 0))
      mixed_tower_1_conv:add(cudnn.SpatialBatchNormalization(config[3][3], std_epsilon, nil, true))
      mixed_tower_1_conv:add(cudnn.ReLU(true))
      mixed_tower_1_conv:add(cudnn.SpatialConvolution(config[3][3], config[3][4], 1, 7, 1, 1, 0, 3))
      mixed_tower_1_conv:add(cudnn.SpatialBatchNormalization(config[3][4], std_epsilon, nil, true))
      mixed_tower_1_conv:add(cudnn.ReLU(true))
      mixed_tower_1_conv:add(cudnn.SpatialConvolution(config[3][4], config[3][5], 7, 1, 1, 1, 3, 0))
      mixed_tower_1_conv:add(cudnn.SpatialBatchNormalization(config[3][5], std_epsilon, nil, true))
      --mixed_tower_1_conv:add(cudnn.ReLU(true))
    local concat_mixed_tower_1_conv = nn.ConcatTable()
      concat_mixed_tower_1_conv:add(mixed_tower_1_conv)
      concat_mixed_tower_1_conv:add(shortcut_mixed_tower_1_conv)
    local add_concat_mixed_tower_1_conv = nn.Sequential()
      add_concat_mixed_tower_1_conv:add(concat_mixed_tower_1_conv)
      add_concat_mixed_tower_1_conv:add(nn.CAddTable())
      add_concat_mixed_tower_1_conv:add(cudnn.ReLU(true))

    local mixed_tower_2_conv_pool = nn.Sequential()
      mixed_tower_2_conv_pool:add(cudnn.SpatialAveragePooling(3, 3, 1, 1, 1, 1))
      mixed_tower_2_conv_pool:add(cudnn.SpatialConvolution(input_size, config[4][1], 1, 1, 1, 1, 0, 0))
      mixed_tower_2_conv_pool:add(cudnn.SpatialBatchNormalization(config[4][1], std_epsilon, nil, true))
      mixed_tower_2_conv_pool:add(cudnn.ReLU(true))
      mixed:add(mixed_conv)
      mixed:add(add_concat_mixed_tower_conv)
      mixed:add(add_concat_mixed_tower_1_conv)
      mixed:add(mixed_tower_2_conv_pool)
  elseif layer_id == 9 then
    -- similar to fig.10
    -- 1x1 -> 3x3(2,valid)
    -- 1x1 -> 7x1 -> 1x7 -> 3x3(2,valid)
    -- 3x3(2,valid,max-pool)
    local shortcut_mixed_conv = nn.Sequential()
      shortcut_mixed_conv:add(cudnn.SpatialConvolution(input_size, config[1][2], 3, 3, 2, 2, 0, 0))
      shortcut_mixed_conv:add(cudnn.SpatialBatchNormalization(config[1][2], std_epsilon, nil, true))

    local mixed_conv = nn.Sequential()
      mixed_conv:add(cudnn.SpatialConvolution(input_size, config[1][1], 1, 1, 1, 1, 0, 0))
      mixed_conv:add(cudnn.SpatialBatchNormalization(config[1][1], std_epsilon, nil, true))
      mixed_conv:add(cudnn.ReLU(true))
      mixed_conv:add(cudnn.SpatialConvolution(config[1][1], config[1][2], 3, 3, 2, 2, 0, 0))
      mixed_conv:add(cudnn.SpatialBatchNormalization(config[1][2], std_epsilon, nil, true))
      --mixed_conv:add(cudnn.ReLU(true))

    local concat_mixed_conv = nn.ConcatTable()
      concat_mixed_conv:add(mixed_conv)
      concat_mixed_conv:add(shortcut_mixed_conv)
    local add_concat_mixed_conv = nn.Sequential()
      add_concat_mixed_conv:add(concat_mixed_conv)
      add_concat_mixed_conv:add(nn.CAddTable())
      add_concat_mixed_conv:add(cudnn.ReLU(true))

    local shortcut_mixed_tower_conv = nn.Sequential()
    shortcut_mixed_tower_conv:add(cudnn.SpatialConvolution(input_size, config[2][4], 3, 3, 2, 2, 0, 0))
    shortcut_mixed_tower_conv:add(cudnn.SpatialBatchNormalization(config[2][4], std_epsilon, nil, true))

    local mixed_tower_conv = nn.Sequential()
      mixed_tower_conv:add(cudnn.SpatialConvolution(input_size, config[2][1], 1, 1, 1, 1, 0, 0))
      mixed_tower_conv:add(cudnn.SpatialBatchNormalization(config[2][1], std_epsilon, nil, true))
      mixed_tower_conv:add(cudnn.ReLU(true))
      mixed_tower_conv:add(cudnn.SpatialConvolution(config[2][1], config[2][2], 7, 1, 1, 1, 3, 0))
      mixed_tower_conv:add(cudnn.SpatialBatchNormalization(config[2][2], std_epsilon, nil, true))
      mixed_tower_conv:add(cudnn.ReLU(true))
      mixed_tower_conv:add(cudnn.SpatialConvolution(config[2][2], config[2][3], 1, 7, 1, 1, 0, 3))
      mixed_tower_conv:add(cudnn.SpatialBatchNormalization(config[2][3], std_epsilon, nil, true))
      mixed_tower_conv:add(cudnn.ReLU(true))
      mixed_tower_conv:add(cudnn.SpatialConvolution(config[2][3], config[2][4], 3, 3, 2, 2, 0, 0))
      mixed_tower_conv:add(cudnn.SpatialBatchNormalization(config[2][4], std_epsilon, nil, true))
      --mixed_tower_conv:add(cudnn.ReLU(true))
    local concat_mixed_tower_conv = nn.ConcatTable()
      concat_mixed_tower_conv:add(mixed_tower_conv)
      concat_mixed_tower_conv:add(shortcut_mixed_tower_conv)
    local add_concat_mixed_tower_conv = nn.Sequential()
      add_concat_mixed_tower_conv:add(concat_mixed_tower_conv)
      add_concat_mixed_tower_conv:add(nn.CAddTable())
      add_concat_mixed_tower_conv:add(cudnn.ReLU(true))

    local mixed_pool = nn.Sequential()
      mixed_pool:add(cudnn.SpatialMaxPooling(3, 3, 2, 2, 0, 0))
      mixed:add(add_concat_mixed_conv)
      mixed:add(add_concat_mixed_tower_conv)
      mixed:add(mixed_pool)
  elseif layer_id == 10 or layer_id == 11 then
    -- fig. 7
    -- 1x1
    -- 1x1 -> 3x1
    --     -> 1x3 
    -- 1x1 -> 3x3(1,same) -> 3x1
    --                    -> 1x3
    -- 3x3(1,same,avg-pool) -> 1x1
    local mixed_conv = nn.Sequential()
      mixed_conv:add(cudnn.SpatialConvolution(input_size, config[1][1], 1, 1, 1, 1, 0, 0))
      mixed_conv:add(cudnn.SpatialBatchNormalization(config[1][1], std_epsilon, nil, true))
      mixed_conv:add(cudnn.ReLU(true))

    local shortcut_mixed_tower_conv = nn.Sequential()
      shortcut_mixed_tower_conv:add(cudnn.SpatialConvolution(input_size, config[2][2]+config[2][3], 1, 1, 1, 1, 0, 0))
      shortcut_mixed_tower_conv:add(cudnn.SpatialBatchNormalization(config[2][2]+config[2][3], std_epsilon, nil, true))

    local mixed_tower_conv = nn.Sequential()
      mixed_tower_conv:add(cudnn.SpatialConvolution(input_size, config[2][1], 1, 1, 1, 1, 0, 0))
      mixed_tower_conv:add(cudnn.SpatialBatchNormalization(config[2][1], std_epsilon, nil, true))
      mixed_tower_conv:add(cudnn.ReLU(true))
      local mixed_tower_expand = nn.Sequential()
        local mixed_expand = nn.DepthConcat(2)
        local mixed_expand_1 = nn.Sequential()
          mixed_expand_1:add(cudnn.SpatialConvolution(config[2][1], config[2][2], 3, 1, 1, 1, 1, 0))
          -- in backward pass, gradInput for cudnn.Spatial-BN is not contiguous
          -- so that we used nn.Spatial-BN.
          mixed_expand_1:add(nn.SpatialBatchNormalization(config[2][2], std_epsilon, nil, true))
          --mixed_expand_1:add(cudnn.ReLU(true))
        local mixed_expand_2 = nn.Sequential()
          mixed_expand_2:add(cudnn.SpatialConvolution(config[2][1], config[2][3], 1, 3, 1, 1, 0, 1))
          -- in backward pass, gradInput for cudnn.Spatial-BN is not contiguous
          -- so that we used nn.Spatial-BN.
          mixed_expand_2:add(nn.SpatialBatchNormalization(config[2][3], std_epsilon, nil, true))
          --mixed_expand_2:add(cudnn.ReLU(true))
        mixed_expand:add(mixed_expand_1)
        mixed_expand:add(mixed_expand_2)
        mixed_tower_expand:add(mixed_expand)
        mixed_tower_conv:add(mixed_tower_expand)
 
    local concat_mixed_tower_conv = nn.ConcatTable()
      concat_mixed_tower_conv:add(mixed_tower_conv)
      concat_mixed_tower_conv:add(shortcut_mixed_tower_conv)

    local add_concat_mixed_tower_conv = nn.Sequential()
    add_concat_mixed_tower_conv:add(concat_mixed_tower_conv)
    add_concat_mixed_tower_conv:add(nn.CAddTable())
    add_concat_mixed_tower_conv:add(cudnn.ReLU(true))

    local shortcut_mixed_tower_1_conv = nn.Sequential()
      shortcut_mixed_tower_1_conv:add(cudnn.SpatialConvolution(input_size, config[3][3]+config[3][4], 1, 1, 1, 1, 0, 0))
      shortcut_mixed_tower_1_conv:add(cudnn.SpatialBatchNormalization(config[3][3]+config[3][4], std_epsilon, nil, true))

    local mixed_tower_1_conv = nn.Sequential()
      mixed_tower_1_conv:add(cudnn.SpatialConvolution(input_size, config[3][1], 1, 1, 1, 1, 0, 0))
      mixed_tower_1_conv:add(cudnn.SpatialBatchNormalization(config[3][1], std_epsilon, nil, true))
      mixed_tower_1_conv:add(cudnn.ReLU(true))
      mixed_tower_1_conv:add(cudnn.SpatialConvolution(config[3][1], config[3][2], 3, 3, 1, 1, 1, 1))
      mixed_tower_1_conv:add(cudnn.SpatialBatchNormalization(config[3][2], std_epsilon, nil, true))
      mixed_tower_1_conv:add(cudnn.ReLU(true))
      local mixed_tower_1_expand = nn.Sequential()
      local mixed_1_expand = nn.DepthConcat(2)
      local mixed_1_expand_1 = nn.Sequential()
        mixed_1_expand_1:add(cudnn.SpatialConvolution(config[3][2], config[3][3], 3, 1, 1, 1, 1, 0))
        -- in backward pass, gradInput for cudnn.Spatial-BN is not contiguous
        -- so that we used nn.Spatial-BN.
        mixed_1_expand_1:add(nn.SpatialBatchNormalization(config[3][3], std_epsilon, nil, true))
        --mixed_1_expand_1:add(cudnn.ReLU(true))
      local mixed_1_expand_2 = nn.Sequential()
        mixed_1_expand_2:add(cudnn.SpatialConvolution(config[3][2], config[3][4], 1, 3, 1, 1, 0, 1))
        -- in backward pass, gradInput for cudnn.Spatial-BN is not contiguous
        -- so that we used nn.Spatial-BN.
        mixed_1_expand_2:add(nn.SpatialBatchNormalization(config[3][4], std_epsilon, nil, true))
        --mixed_1_expand_2:add(cudnn.ReLU(true))
        mixed_1_expand:add(mixed_1_expand_1)
        mixed_1_expand:add(mixed_1_expand_2)
      mixed_tower_1_expand:add(mixed_1_expand)
      mixed_tower_1_conv:add(mixed_tower_1_expand)

    local concat_mixed_tower_1_conv = nn.ConcatTable()
      concat_mixed_tower_1_conv:add(mixed_tower_1_conv)
      concat_mixed_tower_1_conv:add(shortcut_mixed_tower_1_conv)
    local add_concat_mixed_tower_1_conv = nn.Sequential()
      add_concat_mixed_tower_1_conv:add(concat_mixed_tower_1_conv)
      add_concat_mixed_tower_1_conv:add(nn.CAddTable())
      add_concat_mixed_tower_1_conv:add(cudnn.ReLU(true))

    local mixed_tower_2_pool = nn.Sequential()
      mixed_tower_2_pool:add(cudnn.SpatialAveragePooling(3, 3, 1, 1, 1, 1))
      mixed_tower_2_pool:add(cudnn.SpatialConvolution(input_size, config[4][1], 1, 1, 1, 1, 0, 0))
      mixed_tower_2_pool:add(cudnn.SpatialBatchNormalization(config[4][1], std_epsilon, nil, true))
      mixed_tower_2_pool:add(cudnn.ReLU(true))
      mixed:add(mixed_conv)
      mixed:add(add_concat_mixed_tower_conv)
      mixed:add(add_concat_mixed_tower_1_conv)
      mixed:add(mixed_tower_2_pool)
  end
  return mixed
end


function inception_v3_elu_module(mixed_id, input_size, config)
  local layer_id = mixed_id
  local mixed = nn.DepthConcat(2)

  if layer_id == 1 or layer_id == 2 or layer_id == 3 then
    -- similar to fig.4,5
    -- 1x1
    -- 1x1 -> 5x5
    -- 1x1 -> 3x3 -> 3x3
    -- 3x3(1,same,avg-pool) -> 1x1
    local mixed_conv = nn.Sequential()
    -- valid
    mixed_conv:add(cudnn.SpatialConvolution(input_size, config[1][1], 1, 1, 1, 1, 0, 0))
    mixed_conv:add(nn.ELU(1.0, true))
    local mixed_tower_conv = nn.Sequential()
    mixed_tower_conv:add(cudnn.SpatialConvolution(input_size, config[2][1], 1, 1, 1, 1, 0, 0))
    mixed_tower_conv:add(nn.ELU(1.0, true))
    mixed_tower_conv:add(cudnn.SpatialConvolution(config[2][1], config[2][2], 5, 5, 1, 1, 2, 2))
    mixed_tower_conv:add(nn.ELU(1.0, true))
    local mixed_tower_1_conv = nn.Sequential()
    mixed_tower_1_conv:add(cudnn.SpatialConvolution(input_size, config[3][1], 1, 1, 1, 1, 0, 0))
    mixed_tower_1_conv:add(nn.ELU(1.0, true))
    mixed_tower_1_conv:add(cudnn.SpatialConvolution(config[3][1], config[3][2], 3, 3, 1, 1, 1, 1))
    mixed_tower_1_conv:add(nn.ELU(1.0, true))
    mixed_tower_1_conv:add(cudnn.SpatialConvolution(config[3][2], config[3][3], 3, 3, 1, 1, 1, 1))
    mixed_tower_1_conv:add(nn.ELU(1.0, true))
    local mixed_tower_2_conv_pool = nn.Sequential()
    mixed_tower_2_conv_pool:add(cudnn.SpatialAveragePooling(3, 3, 1, 1, 1, 1))
    mixed_tower_2_conv_pool:add(cudnn.SpatialConvolution(input_size, config[4][1], 1, 1, 1, 1, 0, 0))
    mixed_tower_2_conv_pool:add(nn.ELU(1.0, true))
    mixed:add(mixed_conv)
    mixed:add(mixed_tower_conv)
    mixed:add(mixed_tower_1_conv)
    mixed:add(mixed_tower_2_conv_pool)
  elseif layer_id == 4 then
    -- similar to fig.10
    -- 3x3(2,valid)
    -- 1x1 -> 3x3(1,same) -> 3x3(2,valid)
    -- 3x3(2,valid,max-pool)
    local mixed_conv = nn.Sequential()
    mixed_conv:add(cudnn.SpatialConvolution(input_size, config[1][1], 3, 3, 2, 2, 0, 0))
    mixed_conv:add(nn.ELU(1.0, true))
    local mixed_tower_1_conv = nn.Sequential()
    mixed_tower_1_conv:add(cudnn.SpatialConvolution(input_size, config[2][1], 1, 1, 1, 1, 0, 0))
    mixed_tower_1_conv:add(nn.ELU(1.0, true))
    mixed_tower_1_conv:add(cudnn.SpatialConvolution(config[2][1], config[2][2], 3, 3, 1, 1, 1, 1))
    mixed_tower_1_conv:add(nn.ELU(1.0, true))
    mixed_tower_1_conv:add(cudnn.SpatialConvolution(config[2][2], config[2][3], 3, 3, 2, 2, 0, 0))
    mixed_tower_1_conv:add(nn.ELU(1.0, true))
    local mixed_pool = nn.Sequential()
    mixed_pool:add(cudnn.SpatialMaxPooling(3, 3, 2, 2, 0, 0))
    mixed:add(mixed_conv)
    mixed:add(mixed_tower_1_conv)
    mixed:add(mixed_pool)
  elseif layer_id == 5 or layer_id == 6 or layer_id == 7 or layer_id == 8 then
    -- figure 6
    -- 1x1
    -- 1x1 -> 7x1 -> 1x7
    -- 1x1 -> 1x7 -> 7x1 -> 1x7 -> 7x1
    -- 3x3(1,same,avg=pool) -> 1x1
    local mixed_conv = nn.Sequential()
    mixed_conv:add(cudnn.SpatialConvolution(input_size, config[1][1], 1, 1, 1, 1, 0, 0))
    mixed_conv:add(nn.ELU(1.0, true))
    local mixed_tower_conv = nn.Sequential()
    mixed_tower_conv:add(cudnn.SpatialConvolution(input_size, config[2][1], 1, 1, 1, 1, 0, 0))
    mixed_tower_conv:add(nn.ELU(1.0, true))
    mixed_tower_conv:add(cudnn.SpatialConvolution(config[2][1], config[2][2], 7, 1, 1, 1, 3, 0))
    mixed_tower_conv:add(nn.ELU(1.0, true))
    mixed_tower_conv:add(cudnn.SpatialConvolution(config[2][2], config[2][3], 1, 7, 1, 1, 0, 3))
    mixed_tower_conv:add(nn.ELU(1.0, true))
    local mixed_tower_1_conv = nn.Sequential()
    mixed_tower_1_conv:add(cudnn.SpatialConvolution(input_size, config[3][1], 1, 1, 1, 1, 0, 0))
    mixed_tower_1_conv:add(nn.ELU(1.0, true))
    mixed_tower_1_conv:add(cudnn.SpatialConvolution(config[3][1], config[3][2], 1, 7, 1, 1, 0, 3))
    mixed_tower_1_conv:add(nn.ELU(1.0, true))
    mixed_tower_1_conv:add(cudnn.SpatialConvolution(config[3][2], config[3][3], 7, 1, 1, 1, 3, 0))
    mixed_tower_1_conv:add(nn.ELU(1.0, true))
    mixed_tower_1_conv:add(cudnn.SpatialConvolution(config[3][3], config[3][4], 1, 7, 1, 1, 0, 3))
    mixed_tower_1_conv:add(nn.ELU(1.0, true))
    mixed_tower_1_conv:add(cudnn.SpatialConvolution(config[3][4], config[3][5], 7, 1, 1, 1, 3, 0))
    mixed_tower_1_conv:add(nn.ELU(1.0, true))
    local mixed_tower_2_conv_pool = nn.Sequential()
    mixed_tower_2_conv_pool:add(cudnn.SpatialAveragePooling(3, 3, 1, 1, 1, 1))
    mixed_tower_2_conv_pool:add(cudnn.SpatialConvolution(input_size, config[4][1], 1, 1, 1, 1, 0, 0))
    mixed_tower_2_conv_pool:add(nn.ELU(1.0, true))
    mixed:add(mixed_conv)
    mixed:add(mixed_tower_conv)
    mixed:add(mixed_tower_1_conv)
    mixed:add(mixed_tower_2_conv_pool)
  elseif layer_id == 9 then
    -- similar to fig.10
    -- 1x1 -> 3x3(2,valid)
    -- 1x1 -> 7x1 -> 1x7 -> 3x3(2,valid)
    -- 3x3(2,valid,max-pool)
    local mixed_conv = nn.Sequential()
    mixed_conv:add(cudnn.SpatialConvolution(input_size, config[1][1], 1, 1, 1, 1, 0, 0))
    mixed_conv:add(nn.ELU(1.0, true))
    mixed_conv:add(cudnn.SpatialConvolution(config[1][1], config[1][2], 3, 3, 2, 2, 0, 0))
    mixed_conv:add(nn.ELU(1.0, true))
    local mixed_tower_conv = nn.Sequential()
    mixed_tower_conv:add(cudnn.SpatialConvolution(input_size, config[2][1], 1, 1, 1, 1, 0, 0))
    mixed_tower_conv:add(nn.ELU(1.0, true))
    mixed_tower_conv:add(cudnn.SpatialConvolution(config[2][1], config[2][2], 7, 1, 1, 1, 3, 0))
    mixed_tower_conv:add(nn.ELU(1.0, true))
    mixed_tower_conv:add(cudnn.SpatialConvolution(config[2][2], config[2][3], 1, 7, 1, 1, 0, 3))
    mixed_tower_conv:add(nn.ELU(1.0, true))
    mixed_tower_conv:add(cudnn.SpatialConvolution(config[2][3], config[2][4], 3, 3, 2, 2, 0, 0))
    mixed_tower_conv:add(nn.ELU(1.0, true))
    local mixed_pool = nn.Sequential()
    mixed_pool:add(cudnn.SpatialMaxPooling(3, 3, 2, 2, 0, 0))
    mixed:add(mixed_conv)
    mixed:add(mixed_tower_conv)
    mixed:add(mixed_pool)
  elseif layer_id == 10 or layer_id == 11 then
    -- fig. 7
    -- 1x1
    -- 1x1 -> 3x1
    --     -> 1x3 
    -- 1x1 -> 3x3(1,same) -> 3x1
    --                    -> 1x3
    -- 3x3(1,same,avg-pool) -> 1x1
    local mixed_conv = nn.Sequential()
    mixed_conv:add(cudnn.SpatialConvolution(input_size, config[1][1], 1, 1, 1, 1, 0, 0))
    mixed_conv:add(nn.ELU(1.0, true))
    local mixed_tower_conv = nn.Sequential()
    mixed_tower_conv:add(cudnn.SpatialConvolution(input_size, config[2][1], 1, 1, 1, 1, 0, 0))
    mixed_tower_conv:add(nn.ELU(1.0, true))
    local mixed_tower_expand = nn.Sequential()
    local mixed_expand = nn.DepthConcat(2)
    local mixed_expand_1 = nn.Sequential()
    mixed_expand_1:add(cudnn.SpatialConvolution(config[2][1], config[2][2], 3, 1, 1, 1, 1, 0))
    mixed_expand_1:add(nn.ELU(1.0, true))
    local mixed_expand_2 = nn.Sequential()
    mixed_expand_2:add(cudnn.SpatialConvolution(config[2][1], config[2][3], 1, 3, 1, 1, 0, 1))
    mixed_expand_2:add(nn.ELU(1.0, true))
    mixed_expand:add(mixed_expand_1)
    mixed_expand:add(mixed_expand_2)
    mixed_tower_expand:add(mixed_expand)
    mixed_tower_conv:add(mixed_tower_expand)
    local mixed_tower_1_conv = nn.Sequential()
    mixed_tower_1_conv:add(cudnn.SpatialConvolution(input_size, config[3][1], 1, 1, 1, 1, 0, 0))
    mixed_tower_1_conv:add(nn.ELU(1.0, true))
    mixed_tower_1_conv:add(cudnn.SpatialConvolution(config[3][1], config[3][2], 3, 3, 1, 1, 1, 1))
    mixed_tower_1_conv:add(nn.ELU(1.0, true))
    local mixed_tower_1_expand = nn.Sequential()
    local mixed_1_expand = nn.DepthConcat(2)
    local mixed_1_expand_1 = nn.Sequential()
    mixed_1_expand_1:add(cudnn.SpatialConvolution(config[3][2], config[3][3], 3, 1, 1, 1, 1, 0))
    mixed_1_expand_1:add(nn.ELU(1.0, true))
    local mixed_1_expand_2 = nn.Sequential()
    mixed_1_expand_2:add(cudnn.SpatialConvolution(config[3][2], config[3][4], 1, 3, 1, 1, 0, 1))
    mixed_1_expand_2:add(nn.ELU(1.0, true))
    mixed_1_expand:add(mixed_1_expand_1)
    mixed_1_expand:add(mixed_1_expand_2)
    mixed_tower_1_expand:add(mixed_1_expand)
    mixed_tower_1_conv:add(mixed_tower_1_expand)
    local mixed_tower_2_pool = nn.Sequential()
    mixed_tower_2_pool:add(cudnn.SpatialAveragePooling(3, 3, 1, 1, 1, 1))
    mixed_tower_2_pool:add(cudnn.SpatialConvolution(input_size, config[4][1], 1, 1, 1, 1, 0, 0))
    mixed_tower_2_pool:add(nn.ELU(1.0, true))
    mixed:add(mixed_conv)
    mixed:add(mixed_tower_conv)
    mixed:add(mixed_tower_1_conv)
    mixed:add(mixed_tower_2_pool)
  end
  return mixed
end


--  Rethinking the Inception Architecture for Computer Vision, arXiv, 2015
function inception_v3_module(mixed_id, input_size, config)
  local std_epsilon = 0.0010000000475
  local layer_id = mixed_id
  local mixed = nn.DepthConcat(2)

  if layer_id == 1 or layer_id == 2 or layer_id == 3 then
    -- similar to fig.4,5
    -- 1x1
    -- 1x1 -> 5x5
    -- 1x1 -> 3x3 -> 3x3
    -- 3x3(1,same,avg-pool) -> 1x1
    local mixed_conv = nn.Sequential()
    -- valid
    mixed_conv:add(cudnn.SpatialConvolution(input_size, config[1][1], 1, 1, 1, 1, 0, 0))
    mixed_conv:add(cudnn.SpatialBatchNormalization(config[1][1], std_epsilon, nil, true))
    mixed_conv:add(cudnn.ReLU(true))
    local mixed_tower_conv = nn.Sequential()
    mixed_tower_conv:add(cudnn.SpatialConvolution(input_size, config[2][1], 1, 1, 1, 1, 0, 0))
    mixed_tower_conv:add(cudnn.SpatialBatchNormalization(config[2][1], std_epsilon, nil, true))
    mixed_tower_conv:add(cudnn.ReLU(true))
    mixed_tower_conv:add(cudnn.SpatialConvolution(config[2][1], config[2][2], 5, 5, 1, 1, 2, 2))
    mixed_tower_conv:add(cudnn.SpatialBatchNormalization(config[2][2], std_epsilon, nil, true))
    mixed_tower_conv:add(cudnn.ReLU(true))
    local mixed_tower_1_conv = nn.Sequential()
    mixed_tower_1_conv:add(cudnn.SpatialConvolution(input_size, config[3][1], 1, 1, 1, 1, 0, 0))
    mixed_tower_1_conv:add(cudnn.SpatialBatchNormalization(config[3][1], std_epsilon, nil, true))
    mixed_tower_1_conv:add(cudnn.ReLU(true))
    mixed_tower_1_conv:add(cudnn.SpatialConvolution(config[3][1], config[3][2], 3, 3, 1, 1, 1, 1))
    mixed_tower_1_conv:add(cudnn.SpatialBatchNormalization(config[3][2], std_epsilon, nil, true))
    mixed_tower_1_conv:add(cudnn.ReLU(true))
    mixed_tower_1_conv:add(cudnn.SpatialConvolution(config[3][2], config[3][3], 3, 3, 1, 1, 1, 1))
    mixed_tower_1_conv:add(cudnn.SpatialBatchNormalization(config[3][3], std_epsilon, nil, true))
    mixed_tower_1_conv:add(cudnn.ReLU(true))
    local mixed_tower_2_conv_pool = nn.Sequential()
    mixed_tower_2_conv_pool:add(cudnn.SpatialAveragePooling(3, 3, 1, 1, 1, 1))
    mixed_tower_2_conv_pool:add(cudnn.SpatialConvolution(input_size, config[4][1], 1, 1, 1, 1, 0, 0))
    mixed_tower_2_conv_pool:add(cudnn.SpatialBatchNormalization(config[4][1], std_epsilon, nil, true))
    mixed_tower_2_conv_pool:add(cudnn.ReLU(true))
    mixed:add(mixed_conv)
    mixed:add(mixed_tower_conv)
    mixed:add(mixed_tower_1_conv)
    mixed:add(mixed_tower_2_conv_pool)
  elseif layer_id == 4 then
    -- similar to fig.10
    -- 3x3(2,valid)
    -- 1x1 -> 3x3(1,same) -> 3x3(2,valid)
    -- 3x3(2,valid,max-pool)
    local mixed_conv = nn.Sequential()
    mixed_conv:add(cudnn.SpatialConvolution(input_size, config[1][1], 3, 3, 2, 2, 0, 0))
    mixed_conv:add(cudnn.SpatialBatchNormalization(config[1][1], std_epsilon, nil, true))
    mixed_conv:add(cudnn.ReLU(true))
    local mixed_tower_1_conv = nn.Sequential()
    mixed_tower_1_conv:add(cudnn.SpatialConvolution(input_size, config[2][1], 1, 1, 1, 1, 0, 0))
    mixed_tower_1_conv:add(cudnn.SpatialBatchNormalization(config[2][1], std_epsilon, nil, true))
    mixed_tower_1_conv:add(cudnn.ReLU(true))
    mixed_tower_1_conv:add(cudnn.SpatialConvolution(config[2][1], config[2][2], 3, 3, 1, 1, 1, 1))
    mixed_tower_1_conv:add(cudnn.SpatialBatchNormalization(config[2][2], std_epsilon, nil, true))
    mixed_tower_1_conv:add(cudnn.ReLU(true))
    mixed_tower_1_conv:add(cudnn.SpatialConvolution(config[2][2], config[2][3], 3, 3, 2, 2, 0, 0))
    mixed_tower_1_conv:add(cudnn.SpatialBatchNormalization(config[2][3], std_epsilon, nil, true))
    mixed_tower_1_conv:add(cudnn.ReLU(true))
    local mixed_pool = nn.Sequential()
    mixed_pool:add(cudnn.SpatialMaxPooling(3, 3, 2, 2, 0, 0))
    mixed:add(mixed_conv)
    mixed:add(mixed_tower_1_conv)
    mixed:add(mixed_pool)
  elseif layer_id == 5 or layer_id == 6 or layer_id == 7 or layer_id == 8 then
    -- figure 6
    -- 1x1
    -- 1x1 -> 7x1 -> 1x7
    -- 1x1 -> 1x7 -> 7x1 -> 1x7 -> 7x1
    -- 3x3(1,same,avg=pool) -> 1x1
    local mixed_conv = nn.Sequential()
    mixed_conv:add(cudnn.SpatialConvolution(input_size, config[1][1], 1, 1, 1, 1, 0, 0))
    mixed_conv:add(cudnn.SpatialBatchNormalization(config[1][1], std_epsilon, nil, true))
    mixed_conv:add(cudnn.ReLU(true))
    local mixed_tower_conv = nn.Sequential()
    mixed_tower_conv:add(cudnn.SpatialConvolution(input_size, config[2][1], 1, 1, 1, 1, 0, 0))
    mixed_tower_conv:add(cudnn.SpatialBatchNormalization(config[2][1], std_epsilon, nil, true))
    mixed_tower_conv:add(cudnn.ReLU(true))
    mixed_tower_conv:add(cudnn.SpatialConvolution(config[2][1], config[2][2], 7, 1, 1, 1, 3, 0))
    mixed_tower_conv:add(cudnn.SpatialBatchNormalization(config[2][2], std_epsilon, nil, true))
    mixed_tower_conv:add(cudnn.ReLU(true))
    mixed_tower_conv:add(cudnn.SpatialConvolution(config[2][2], config[2][3], 1, 7, 1, 1, 0, 3))
    mixed_tower_conv:add(cudnn.SpatialBatchNormalization(config[2][3], std_epsilon, nil, true))
    mixed_tower_conv:add(cudnn.ReLU(true))
    local mixed_tower_1_conv = nn.Sequential()
    mixed_tower_1_conv:add(cudnn.SpatialConvolution(input_size, config[3][1], 1, 1, 1, 1, 0, 0))
    mixed_tower_1_conv:add(cudnn.SpatialBatchNormalization(config[3][1], std_epsilon, nil, true))
    mixed_tower_1_conv:add(cudnn.ReLU(true))
    mixed_tower_1_conv:add(cudnn.SpatialConvolution(config[3][1], config[3][2], 1, 7, 1, 1, 0, 3))
    mixed_tower_1_conv:add(cudnn.SpatialBatchNormalization(config[3][2], std_epsilon, nil, true))
    mixed_tower_1_conv:add(cudnn.ReLU(true))
    mixed_tower_1_conv:add(cudnn.SpatialConvolution(config[3][2], config[3][3], 7, 1, 1, 1, 3, 0))
    mixed_tower_1_conv:add(cudnn.SpatialBatchNormalization(config[3][3], std_epsilon, nil, true))
    mixed_tower_1_conv:add(cudnn.ReLU(true))
    mixed_tower_1_conv:add(cudnn.SpatialConvolution(config[3][3], config[3][4], 1, 7, 1, 1, 0, 3))
    mixed_tower_1_conv:add(cudnn.SpatialBatchNormalization(config[3][4], std_epsilon, nil, true))
    mixed_tower_1_conv:add(cudnn.ReLU(true))
    mixed_tower_1_conv:add(cudnn.SpatialConvolution(config[3][4], config[3][5], 7, 1, 1, 1, 3, 0))
    mixed_tower_1_conv:add(cudnn.SpatialBatchNormalization(config[3][5], std_epsilon, nil, true))
    mixed_tower_1_conv:add(cudnn.ReLU(true))
    local mixed_tower_2_conv_pool = nn.Sequential()
    mixed_tower_2_conv_pool:add(cudnn.SpatialAveragePooling(3, 3, 1, 1, 1, 1))
    mixed_tower_2_conv_pool:add(cudnn.SpatialConvolution(input_size, config[4][1], 1, 1, 1, 1, 0, 0))
    mixed_tower_2_conv_pool:add(cudnn.SpatialBatchNormalization(config[4][1], std_epsilon, nil, true))
    mixed_tower_2_conv_pool:add(cudnn.ReLU(true))
    mixed:add(mixed_conv)
    mixed:add(mixed_tower_conv)
    mixed:add(mixed_tower_1_conv)
    mixed:add(mixed_tower_2_conv_pool)
  elseif layer_id == 9 then
    -- similar to fig.10
    -- 1x1 -> 3x3(2,valid)
    -- 1x1 -> 7x1 -> 1x7 -> 3x3(2,valid)
    -- 3x3(2,valid,max-pool)
    local mixed_conv = nn.Sequential()
    mixed_conv:add(cudnn.SpatialConvolution(input_size, config[1][1], 1, 1, 1, 1, 0, 0))
    mixed_conv:add(cudnn.SpatialBatchNormalization(config[1][1], std_epsilon, nil, true))
    mixed_conv:add(cudnn.ReLU(true))
    mixed_conv:add(cudnn.SpatialConvolution(config[1][1], config[1][2], 3, 3, 2, 2, 0, 0))
    mixed_conv:add(cudnn.SpatialBatchNormalization(config[1][2], std_epsilon, nil, true))
    mixed_conv:add(cudnn.ReLU(true))
    local mixed_tower_conv = nn.Sequential()
    mixed_tower_conv:add(cudnn.SpatialConvolution(input_size, config[2][1], 1, 1, 1, 1, 0, 0))
    mixed_tower_conv:add(cudnn.SpatialBatchNormalization(config[2][1], std_epsilon, nil, true))
    mixed_tower_conv:add(cudnn.ReLU(true))
    mixed_tower_conv:add(cudnn.SpatialConvolution(config[2][1], config[2][2], 7, 1, 1, 1, 3, 0))
    mixed_tower_conv:add(cudnn.SpatialBatchNormalization(config[2][2], std_epsilon, nil, true))
    mixed_tower_conv:add(cudnn.ReLU(true))
    mixed_tower_conv:add(cudnn.SpatialConvolution(config[2][2], config[2][3], 1, 7, 1, 1, 0, 3))
    mixed_tower_conv:add(cudnn.SpatialBatchNormalization(config[2][3], std_epsilon, nil, true))
    mixed_tower_conv:add(cudnn.ReLU(true))
    mixed_tower_conv:add(cudnn.SpatialConvolution(config[2][3], config[2][4], 3, 3, 2, 2, 0, 0))
    mixed_tower_conv:add(cudnn.SpatialBatchNormalization(config[2][4], std_epsilon, nil, true))
    mixed_tower_conv:add(cudnn.ReLU(true))
    local mixed_pool = nn.Sequential()
    mixed_pool:add(cudnn.SpatialMaxPooling(3, 3, 2, 2, 0, 0))
    mixed:add(mixed_conv)
    mixed:add(mixed_tower_conv)
    mixed:add(mixed_pool)
  elseif layer_id == 10 or layer_id == 11 then
    -- fig. 7
    -- 1x1
    -- 1x1 -> 3x1
    --     -> 1x3 
    -- 1x1 -> 3x3(1,same) -> 3x1
    --                    -> 1x3
    -- 3x3(1,same,avg-pool) -> 1x1
    local mixed_conv = nn.Sequential()
    mixed_conv:add(cudnn.SpatialConvolution(input_size, config[1][1], 1, 1, 1, 1, 0, 0))
    mixed_conv:add(cudnn.SpatialBatchNormalization(config[1][1], std_epsilon, nil, true))
    mixed_conv:add(cudnn.ReLU(true))
    local mixed_tower_conv = nn.Sequential()
    mixed_tower_conv:add(cudnn.SpatialConvolution(input_size, config[2][1], 1, 1, 1, 1, 0, 0))
    mixed_tower_conv:add(cudnn.SpatialBatchNormalization(config[2][1], std_epsilon, nil, true))
    mixed_tower_conv:add(cudnn.ReLU(true))
    local mixed_tower_expand = nn.Sequential()
    local mixed_expand = nn.DepthConcat(2)
    local mixed_expand_1 = nn.Sequential()
    mixed_expand_1:add(cudnn.SpatialConvolution(config[2][1], config[2][2], 3, 1, 1, 1, 1, 0))
    mixed_expand_1:add(cudnn.SpatialBatchNormalization(config[2][2], std_epsilon, nil, true))
    mixed_expand_1:add(cudnn.ReLU(true))
    local mixed_expand_2 = nn.Sequential()
    mixed_expand_2:add(cudnn.SpatialConvolution(config[2][1], config[2][3], 1, 3, 1, 1, 0, 1))
    mixed_expand_2:add(cudnn.SpatialBatchNormalization(config[2][3], std_epsilon, nil, true))
    mixed_expand_2:add(cudnn.ReLU(true))
    mixed_expand:add(mixed_expand_1)
    mixed_expand:add(mixed_expand_2)
    mixed_tower_expand:add(mixed_expand)
    mixed_tower_conv:add(mixed_tower_expand)
    local mixed_tower_1_conv = nn.Sequential()
    mixed_tower_1_conv:add(cudnn.SpatialConvolution(input_size, config[3][1], 1, 1, 1, 1, 0, 0))
    mixed_tower_1_conv:add(cudnn.SpatialBatchNormalization(config[3][1], std_epsilon, nil, true))
    mixed_tower_1_conv:add(cudnn.ReLU(true))
    mixed_tower_1_conv:add(cudnn.SpatialConvolution(config[3][1], config[3][2], 3, 3, 1, 1, 1, 1))
    mixed_tower_1_conv:add(cudnn.SpatialBatchNormalization(config[3][2], std_epsilon, nil, true))
    mixed_tower_1_conv:add(cudnn.ReLU(true))
    local mixed_tower_1_expand = nn.Sequential()
    local mixed_1_expand = nn.DepthConcat(2)
    local mixed_1_expand_1 = nn.Sequential()
    mixed_1_expand_1:add(cudnn.SpatialConvolution(config[3][2], config[3][3], 3, 1, 1, 1, 1, 0))
    mixed_1_expand_1:add(cudnn.SpatialBatchNormalization(config[3][3], std_epsilon, nil, true))
    mixed_1_expand_1:add(cudnn.ReLU(true))
    local mixed_1_expand_2 = nn.Sequential()
    mixed_1_expand_2:add(cudnn.SpatialConvolution(config[3][2], config[3][4], 1, 3, 1, 1, 0, 1))
    mixed_1_expand_2:add(cudnn.SpatialBatchNormalization(config[3][4], std_epsilon, nil, true))
    mixed_1_expand_2:add(cudnn.ReLU(true))
    mixed_1_expand:add(mixed_1_expand_1)
    mixed_1_expand:add(mixed_1_expand_2)
    mixed_tower_1_expand:add(mixed_1_expand)
    mixed_tower_1_conv:add(mixed_tower_1_expand)
    local mixed_tower_2_pool = nn.Sequential()
    mixed_tower_2_pool:add(cudnn.SpatialAveragePooling(3, 3, 1, 1, 1, 1))
    mixed_tower_2_pool:add(cudnn.SpatialConvolution(input_size, config[4][1], 1, 1, 1, 1, 0, 0))
    mixed_tower_2_pool:add(cudnn.SpatialBatchNormalization(config[4][1], std_epsilon, nil, true))
    mixed_tower_2_pool:add(cudnn.ReLU(true))
    mixed:add(mixed_conv)
    mixed:add(mixed_tower_conv)
    mixed:add(mixed_tower_1_conv)
    mixed:add(mixed_tower_2_pool)
  end
  return mixed
end

