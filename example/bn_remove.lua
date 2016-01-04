require 'torch'
require 'cutorch'
require 'nn'
require 'cunn'
require 'cudnn'
paths.dofile('../utils/net_utils.lua')

--cutorch.setDevice(1)
print '===> Loading model'
local model_filename =
  '/storage/ImageNet/ILSVRC2012/torch_cache/inception7_residual/gpu2_residual_feature_lr0.045_epoch19_Wed_Dec_30_20_40_18_2015/model_35.t7'
  --'/storage/ImageNet/ILSVRC2012/torch_cache/inception7_residual/gpu2_residual_feature_lr0.045_epoch19_Wed_Dec_30_20_40_18_2015/model_31.t7'
  --'/storage/ImageNet/ILSVRC2012/torch_cache/inception7_residual/gpu2_residual_feature_lr0.045_epoch19_Tue_Dec_29_15_06_01_2015/model_29.t7'
  --'/storage/ImageNet/ILSVRC2012/torch_cache/inception7_residual/digits_gpu2_residual_feature_lr0.045SatDec1912:29:402015/model_19.t7'
  --'/storage/product/clothes/torch_cache/inception7/clothes_gpu2_lr0.045_digits_gpu_2_lr0.045SatDec514:08:122015MonDec2117:51:172015/model_11.t7'
  --'/storage/product/clothes/torch_cache/inception7/clothes_gpu2_lr0.045_digits_gpu_2_lr0.045SatDec514:08:122015MonDec2117:51:172015/model_8.t7'
  --'/storage/product/clothes/torch_cache/inception7/clothes_gpu2_lr0.045_digits_gpu_2_lr0.045SatDec514:08:122015MonDec2117:51:172015/model_1.t7'
  --'/storage/ImageNet/ILSVRC2012/torch_cache/inception7/digits_gpu_2_lr0.045SatDec514:08:122015/model_40.t7'
  --'/data2/product/det/torch_cache/inception6/det_stnThuDec318:29:322015/model_29.t7'
local output_model_filename =
  '/storage/ImageNet/ILSVRC2012/torch_cache/inception7_residual/gpu2_residual_feature_lr0.045_epoch19_Wed_Dec_30_20_40_18_2015/model_35.bn_removed.t7'
  --'/storage/ImageNet/ILSVRC2012/torch_cache/inception7_residual/gpu2_residual_feature_lr0.045_epoch19_Wed_Dec_30_20_40_18_2015/model_31.bn_removed.t7'
  --'/storage/ImageNet/ILSVRC2012/torch_cache/inception7_residual/gpu2_residual_feature_lr0.045_epoch19_Tue_Dec_29_15_06_01_2015/model_29.bn_removed.t7'
  --'/storage/ImageNet/ILSVRC2012/torch_cache/inception7_residual/digits_gpu2_residual_feature_lr0.045SatDec1912:29:402015/model_19.bn_removed.t7'
  --'/storage/product/clothes/torch_cache/inception7/clothes_gpu2_lr0.045_digits_gpu_2_lr0.045SatDec514:08:122015MonDec2117:51:172015/model_11.bn_removed.t7'
  --'/storage/product/clothes/torch_cache/inception7/clothes_gpu2_lr0.045_digits_gpu_2_lr0.045SatDec514:08:122015MonDec2117:51:172015/model_8.bn_removed.t7'
  --'/storage/product/clothes/torch_cache/inception7/clothes_gpu2_lr0.045_digits_gpu_2_lr0.045SatDec514:08:122015MonDec2117:51:172015/model_1.bn_removed.t7'
  --'/storage/ImageNet/ILSVRC2012/torch_cache/inception7/digits_gpu_2_lr0.045SatDec514:08:122015/model_40.bn_removed.t7'
  --'/data2/product/det/torch_cache/inception6/det_stnThuDec318:29:322015/model_29.bn_removed.t7'
local original_model = torch.load(model_filename)
local feature_encoder = original_model:get(1):get(1)
local classifier = original_model:get(2)
classifier.modules[#classifier.modules] = nil
classifier:add(cudnn.SoftMax())
local model = nn.Sequential()
model:add(feature_encoder):add(classifier)

--model_bn = paths.dofile('../utils/BN-absorber.lua')(model:clone())
model_bn = paths.dofile('../utils/BN-absorber.lua')(model)
print(model_bn)
print('===> Saveing done for original model: '..model_filename)
save_net(model_bn, output_model_filename)

