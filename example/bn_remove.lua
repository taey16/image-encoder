require 'torch'
require 'cutorch'
require 'nn'
require 'cunn'
require 'cudnn'
paths.dofile('../utils/net_utils.lua')


print '===> Loading model'
local model_filename =
  --'/storage/product/det/torch_cache/det_X_gpu2_resception_epoch1_nag_lr0.01125_decay_seed0.500_start37040_every37040/model_15.t7'
  --'/storage/product/det/torch_cache/det_X_gpu2__epoch1_nag_lr0.01125_decay_seed0.500_start40785_every40785/model_3.t7'
  --'/data2/ImageNet/ILSVRC2012/torch_cache/X_gpu1_resception_nag_lr0.00450_decay_start0_every160000/model_19.t7'
  '/data2/ImageNet/ILSVRC2012/torch_cache/X_gpu1_resception_nag_lr0.00450_decay_start0_every160000/model_29.t7'
  --'/data2/ImageNet/ILSVRC2012/torch_cache/resception/X_gpu1_resception_nag_0.04500_bninit_linearinit_Tue_Feb_16_13_01_55_2016/model_17.t7'
  --'/data2/ImageNet/ILSVRC2012/torch_cache/inception7_residual/digits_gpu1_inception-v3-2015-12-05_lr0.045_Mon_Jan_18_13_23_03_2016/model_33.t7'
  --'/data2/ImageNet/ILSVRC2012/torch_cache/inception7_residual/digits_gpu1_inception-v3-2015-12-05_lr0.045_Mon_Jan_18_13_23_03_2016/model_31.t7'
  --'/data2/ImageNet/ILSVRC2012/torch_cache/inception-v3-2015-12-05/digits_gpu2_inception-v3-2015-12-05_Sat_Jan_30_17_16_06_2016/model_16.t7'
  --'/data2/ImageNet/ILSVRC2012/torch_cache/inception-v3-2015-12-05/digits_gpu2_inception-v3-2015-12-05_Wed_Jan_27_22_47_34_2016/model_10.t7'
  --'/data2/ImageNet/ILSVRC2012/torch_cache/inception-v3-2015-12-05/digits_gpu2_inception-v3-2015-12-05_Wed_Jan_27_22_47_34_2016/model_9.t7'
  --'/data2/ImageNet/ILSVRC2012/torch_cache/inception-v3-2015-12-05/digits_gpu2_inception-v3-2015-12-05_Thu_Jan_21_08_48_49_2016/model_8.t7'
  --'/data2/ImageNet/ILSVRC2012/torch_cache/inception-v3-2015-12-05/digits_gpu2_inception-v3-2015-12-05_Thu_Jan_21_08_48_49_2016/model_6.t7'
  --'/storage/ImageNet/ILSVRC2012/torch_cache/inception7_residual/gpu2_residual_feature_lr0.045_epoch19_Wed_Dec_30_20_40_18_2015/model_35.t7'
  --'/storage/ImageNet/ILSVRC2012/torch_cache/inception7_residual/gpu2_residual_feature_lr0.045_epoch19_Wed_Dec_30_20_40_18_2015/model_31.t7'
  --'/storage/ImageNet/ILSVRC2012/torch_cache/inception7_residual/gpu2_residual_feature_lr0.045_epoch19_Tue_Dec_29_15_06_01_2015/model_29.t7'
  --'/storage/ImageNet/ILSVRC2012/torch_cache/inception7_residual/digits_gpu2_residual_feature_lr0.045SatDec1912:29:402015/model_19.t7'
  --'/storage/product/clothes/torch_cache/inception7/clothes_gpu2_lr0.045_digits_gpu_2_lr0.045SatDec514:08:122015MonDec2117:51:172015/model_11.t7'
  --'/storage/product/clothes/torch_cache/inception7/clothes_gpu2_lr0.045_digits_gpu_2_lr0.045SatDec514:08:122015MonDec2117:51:172015/model_8.t7'
  --'/storage/product/clothes/torch_cache/inception7/clothes_gpu2_lr0.045_digits_gpu_2_lr0.045SatDec514:08:122015MonDec2117:51:172015/model_1.t7'
  --'/storage/ImageNet/ILSVRC2012/torch_cache/inception7/digits_gpu_2_lr0.045SatDec514:08:122015/model_40.t7'
  --'/data2/product/det/torch_cache/inception6/det_stnThuDec318:29:322015/model_29.t7'
local output_model_filename =
  --'/storage/product/det/torch_cache/det_X_gpu2_resception_epoch1_nag_lr0.01125_decay_seed0.500_start37040_every37040/model_15.bn_removed.t7'
  --'/storage/product/det/torch_cache/det_X_gpu2__epoch1_nag_lr0.01125_decay_seed0.500_start40785_every40785/model_3.bn_removed.t7'
  --'/data2/ImageNet/ILSVRC2012/torch_cache/X_gpu1_resception_nag_lr0.00450_decay_start0_every160000/model_19.bn_removed.t7'
  '/data2/ImageNet/ILSVRC2012/torch_cache/X_gpu1_resception_nag_lr0.00450_decay_start0_every160000/model_29.bn_removed.t7'
  --'/data2/ImageNet/ILSVRC2012/torch_cache/resception/X_gpu1_resception_nag_0.04500_bninit_linearinit_Tue_Feb_16_13_01_55_2016/model_17.bn_removed.t7'
  --'/data2/ImageNet/ILSVRC2012/torch_cache/inception7_residual/digits_gpu1_inception-v3-2015-12-05_lr0.045_Mon_Jan_18_13_23_03_2016/model_33.bn_removed.t7'
  --'/data2/ImageNet/ILSVRC2012/torch_cache/inception7_residual/digits_gpu1_inception-v3-2015-12-05_lr0.045_Mon_Jan_18_13_23_03_2016/model_31.bn_removed.t7'
  --'/data2/ImageNet/ILSVRC2012/torch_cache/inception-v3-2015-12-05/digits_gpu2_inception-v3-2015-12-05_Sat_Jan_30_17_16_06_2016/model_16.bn_removed.t7'
  --'/data2/ImageNet/ILSVRC2012/torch_cache/inception-v3-2015-12-05/digits_gpu2_inception-v3-2015-12-05_Wed_Jan_27_22_47_34_2016/model_10.bn_removed.t7'
  --'/data2/ImageNet/ILSVRC2012/torch_cache/inception-v3-2015-12-05/digits_gpu2_inception-v3-2015-12-05_Wed_Jan_27_22_47_34_2016/model_9.bn_removed.t7'
  --'/data2/ImageNet/ILSVRC2012/torch_cache/inception-v3-2015-12-05/digits_gpu2_inception-v3-2015-12-05_Thu_Jan_21_08_48_49_2016/model_8.bn_removed.t7'
  --'/data2/ImageNet/ILSVRC2012/torch_cache/inception-v3-2015-12-05/digits_gpu2_inception-v3-2015-12-05_Thu_Jan_21_08_48_49_2016/model_6.bn_removed.t7'
  --'/storage/ImageNet/ILSVRC2012/torch_cache/inception7_residual/gpu2_residual_feature_lr0.045_epoch19_Wed_Dec_30_20_40_18_2015/model_35.bn_removed.t7'
  --'/storage/ImageNet/ILSVRC2012/torch_cache/inception7_residual/gpu2_residual_feature_lr0.045_epoch19_Wed_Dec_30_20_40_18_2015/model_31.bn_removed.t7'
  --'/storage/ImageNet/ILSVRC2012/torch_cache/inception7_residual/gpu2_residual_feature_lr0.045_epoch19_Tue_Dec_29_15_06_01_2015/model_29.bn_removed.t7'
  --'/storage/ImageNet/ILSVRC2012/torch_cache/inception7_residual/digits_gpu2_residual_feature_lr0.045SatDec1912:29:402015/model_19.bn_removed.t7'
  --'/storage/product/clothes/torch_cache/inception7/clothes_gpu2_lr0.045_digits_gpu_2_lr0.045SatDec514:08:122015MonDec2117:51:172015/model_11.bn_removed.t7'
  --'/storage/product/clothes/torch_cache/inception7/clothes_gpu2_lr0.045_digits_gpu_2_lr0.045SatDec514:08:122015MonDec2117:51:172015/model_8.bn_removed.t7'
  --'/storage/product/clothes/torch_cache/inception7/clothes_gpu2_lr0.045_digits_gpu_2_lr0.045SatDec514:08:122015MonDec2117:51:172015/model_1.bn_removed.t7'
  --'/storage/ImageNet/ILSVRC2012/torch_cache/inception7/digits_gpu_2_lr0.045SatDec514:08:122015/model_40.bn_removed.t7'
  --'/data2/product/det/torch_cache/inception6/det_stnThuDec318:29:322015/model_29.bn_removed.t7'
local model = torch.load(model_filename)
print(model)
model.modules[#model.modules] = nil
model:add(cudnn.SoftMax())
print(model)

--model_bn = paths.dofile('../utils/BN-absorber.lua')(model:clone())
model_bn = paths.dofile('../utils/BN-absorber.lua')(model)
print(model_bn)
print('===> Saveing for original model: '..model_filename)
save_net(model_bn, output_model_filename)

