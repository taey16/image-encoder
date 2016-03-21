require 'torch'
require 'nn'
require 'nngraph'
require 'cutorch'
require 'cunn'
require 'cudnn'
local cjson = require 'cjson'
package.path = '/works/vision_language/?.lua;' .. package.path
require 'misc.DataLoaderRaw'
require 'models.LanguageModel'
require 'models.FeatExpander'
local net_utils = require 'misc.net_utils'


local model_filename = 
  '/storage/attribute/checkpoints/tshirts_shirts_blous_knit_97683_12800_seq_length-1/_resception_bn_removed_epoch19_bs16_flipfalse_croptrue_lstm_tanh_hidden384_layer2_dropout0.5_lr1.000000e-03_anneal_seed0.94_start53050_every5305_finetune-1_cnnlr1.000000e-03_cnnwc1.000000e-04/model_id_resception_bn_removed_epoch19_bs16_flipfalse_croptrue_lstm_tanh_hidden384_layer2_dropout0.5_lr1.000000e-03_anneal_seed0.94_start53050_every5305_finetune-1_cnnlr1.000000e-03_cnnwc1.000000e-04.t7'
  --'/storage/attribute/checkpoints/tshirts_shirts_blous_knit_103607_8000_seq_length-1/_resception_epoch19_bs16_flipfalse_croptrue_lstm_tanh_hidden256_layer2_dropout0.5_lr1.000000e-03_anneal_seed0.50_start50000_every25000_finetune0_cnnlr1.000000e-03_test/model_id_resception_epoch19_bs16_flipfalse_croptrue_lstm_tanh_hidden256_layer2_dropout0.5_lr1.000000e-03_anneal_seed0.50_start50000_every25000_finetune0_cnnlr1.000000e-03_test.t7'
  --'/storage/coco/checkpoints/_inception-v3-2015-12-05_bn_removed_epoch10_mean_std_modified_bs16_embedding2048_encode384_layer3_lr4e-4/model_id_inception-v3-2015-12-05_bn_removed_epoch10_mean_std_modified_bs16_embedding2048_encode384_layer3_lr4e-4.t7'
  --'/storage/coco/checkpoints/_ReCept_bn_removed_epoch35_bs16_embedding2048_encode384_layer2_lr4e-4/model_id_ReCept_bn_removed_epoch35_bs16_embedding2048_encode384_layer2_lr4e-4.t7'
  --'/storage/coco/checkpoints/_ReCept_bn_removed_epoch35_bs16_embedding2048_encode256_layer2_lr4e-4/model_id_ReCept_bn_removed_epoch35_bs16_embedding2048_encode256_layer2_lr4e-4.t7'
  --'/storage/coco/checkpoints/_inception7_bs16_encode256_layer2/model_id_inception7_bs16_encode256_layer2.t7'
  --'/storage/coco/checkpoints/_inception7_bs16_encode512/model_id_inception7_bs16_encode512.t7'
  --'/storage/coco/checkpoints/_inception7_bs16_encode512_finetune_lr4e-6_clr1e-7_wc2e-5/model_id_inception7_bs16_encode512_finetune_lr4e-6_clr1e-7_wc2e-5.t7'
local checkpoint = torch.load(model_filename)
local batch_size = checkpoint.opt.batch_size
local opt = {
  'rnn_size', 
  'input_encoding_size', 
  'drop_prob_lm', 
  'cnn_proto', 
  'cnn_model', 
  'seq_per_img', 
  'image_size', 
  'crop_size'
}
for k,v in pairs(opt) do 
  opt[v] = checkpoint.opt[v]
end
local vocab = checkpoint.vocab
local sample_opts = { 
  sample_max = opt.sample_max, 
  beam_size = 10, 
  temperature = opt.temperature 
}

local protos = checkpoint.protos
protos.expander = nn.FeatExpander(opt.seq_per_img)
protos.lm:createClones()
for k,v in pairs(protos) do v:cuda() end
protos.cnn:evaluate()
protos.lm:evaluate()

--[[
for k, v in ipairs(info) do
  local img = image.load(fname)
  img = image.scale(img, opt.image_size, opt.image_size)
  if img:size(1) == 1 then
    img = img:view(1,img:size(2), img:size(3)):repeatTensor(3,1,1)
  end
  img = net_utils.preprocess_inception7_predict(img, opt.crop_size, false, 1)
  local data = torch.CudaTensor(2, 3, opt.crop_size, opt.crop_size):fill(0)
  data[{{1},{},{},{}}] = img
  local feats = protos.cnn:forward(data)
  local seq = protos.lm:sample(feats, sample_opts)
  local sents = net_utils.decode_sequence(vocab, seq)
end
--]]
