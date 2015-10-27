
-- global
-- opt
-- model
-- criterion
paths.dofile('models/init_model_weight.lua')
paths.dofile(opt.model_script)

print('===> Creating model from script: ' .. opt.model_script)
model = createModel()

spanet = {}
--spanet_lay17 = {}
if opt.use_stn then 
  spanet = paths.dofile('models/spatial_transformer.lua')
  --spanet_lay17 = paths.dofile('model/spatial_transformer_1.lua')
  model:insert(spanet,1)
  --model:insert(spanet_lay17,17)
end

MSRinit(model)
criterion = nn.ClassNLLCriterion()

if opt.model_filename then
  model = torch.load(opt.model_filename)
  print('===> Loaded model from file: ' .. opt.model_filename)
end

if opt.use_cuda then
  model:cuda()
  criterion:cuda()
  print('===> model:cuda()')
end

print(model)
print(criterion)

