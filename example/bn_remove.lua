require 'torch'
require 'cutorch'
require 'nn'
require 'cunn'
require 'cudnn'
paths.dofile('../utils/net_utils.lua')


print '===> Loading model'
local encoder_model_filename =
  '/data2/ImageNet/ILSVRC2012/torch_cache/X_gpu1_resception_nag_lr0.00450_decay_start0_every160000/model_29.bn_removed.t7'
  --'/data2/ImageNet/ILSVRC2012/torch_cache/X_gpu1_resception_nag_lr0.00450_decay_start0_every160000/model_19.bn_removed.t7'
local classifier_model_filename = 
  -- long line
  '/storage/freebee/attribute_long_line/torch_cache/devfalse_attribute_long_line_X_gpu2_resception_epoch1_stratified_samplefalse_nag_lr0.10000_decay_seed0.940_start0_every2653/model_66.t7'
  -- slit collar
  --'/storage/freebee/attribute_slit_collar/torch_cache/devfalse_attribute_slit_collar_X_gpu2_resception_epoch1_stratified_samplefalse_nag_lr0.10000_decay_seed0.940_start0_every2837/model_59.t7'
  -- china_collar
  --'/storage/freebee/attribute_china_collar/torch_cache/devtrue_attribute_china_collar_X_gpu2_resception_epoch1_stratified_samplefalse_nag_lr0.10000_decay_seed0.940_start0_every2837/model_109.t7'
  -- button
  --'/storage/freebee/attribute_button/torch_cache/devfalse_attribute_button_X_gpu2_resception_epoch1_stratified_samplefalse_nag_lr0.10000_decay_seed0.940_start0_every2837/model_37.t7'
local output_model_filename =
  -- long line
  '/storage/freebee/attribute_long_line/torch_cache/devfalse_attribute_long_line_X_gpu2_resception_epoch1_stratified_samplefalse_nag_lr0.10000_decay_seed0.940_start0_every2653/model_reception_29.bn_removed.classifier_66.bn_removed.t7'
  -- slit collar
  --'/storage/freebee/attribute_slit_collar/torch_cache/devfalse_attribute_slit_collar_X_gpu2_resception_epoch1_stratified_samplefalse_nag_lr0.10000_decay_seed0.940_start0_every2837/model_reception_19.bn_removed.classifier_59.bn_removed.t7'
  -- china_collar
  --'/storage/freebee/attribute_china_collar/torch_cache/devtrue_attribute_china_collar_X_gpu2_resception_epoch1_stratified_samplefalse_nag_lr0.10000_decay_seed0.940_start0_every2837/model_reception_19.bn_removed.classifier_109.bn_removed.t7'
  -- button
  --'/storage/freebee/attribute_button/torch_cache/devfalse_attribute_button_X_gpu2_resception_epoch1_stratified_samplefalse_nag_lr0.10000_decay_seed0.940_start0_every2837/model_reception_29.bn_removed.classifier_37.bn_removed.t7'
local encoder = torch.load(encoder_model_filename)
local classifier= torch.load(classifier_model_filename)
encoder:remove(#encoder.modules)
encoder:remove(#encoder.modules)
classifier:remove(#classifier.modules)
classifier:add(cudnn.SoftMax())
collectgarbage()
local model = nn.Sequential()
model:add(encoder):add(classifier)
collectgarbage()
print(model)

model_bn = paths.dofile('../utils/BN-absorber.lua')(model)
print(model_bn)
print('===> Saveing for original model: '..classifier_model_filename)
save_net(model_bn, output_model_filename)
