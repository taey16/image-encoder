require 'torch'
require 'cutorch'
require 'nn'
require 'cunn'
require 'cudnn'
paths.dofile('../utils/net_utils.lua')

--cutorch.setDevice(1)
print '===> Loading model'
local encoder_model_filename =
  '/data2/ImageNet/ILSVRC2012/torch_cache/X_gpu1_resception_nag_lr0.00450_decay_start0_every160000/model_19.bn_removed.t7'
local classifier_model_filename = 
  '/storage/freebee/attribute_button/torch_cache/devfalse_attribute_button_X_gpu2_resception_epoch1_stratified_samplefalse_nag_lr0.10000_decay_seed0.940_start0_every2837/model_79.t7'
local output_model_filename =
  '/storage/freebee/attribute_button/torch_cache/devfalse_attribute_button_X_gpu2_resception_epoch1_stratified_samplefalse_nag_lr0.10000_decay_seed0.940_start0_every2837/model_reception_19.bn_removed.classifier_79.bn_removed..t7'
local encoder = torch.load(encoder_model_filename)
local classifier= torch.load(classifier_model_filename)
encoder.modules[#encoder.modules] = nil
encoder.modules[#encoder.modules] = nil
for i=1,7 do
  encoder:add(classifier:get(i))
end
encoder:add(cudnn.SoftMax())
local model = encoder
collectgarbage()
print(model)

model_bn = paths.dofile('../utils/BN-absorber.lua')(model)
print(model_bn)
print('===> Saveing for original model: '..classifier_model_filename)
save_net(model_bn, output_model_filename)
