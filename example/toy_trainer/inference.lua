
require 'torch'
require 'cudnn'
require 'cunn'
require 'stn'

torch.setdefaulttensortype('torch.FloatTensor')

model = torch.load('model_3.t7')
data = torch.load('/storage/mnist/mnist.t7/test_32x32.t7','ascii')
test.data = data.data:float()
test.labels=data.labels:float()
print(test.labels:nDimension())
input = torch.Tensor(64,1,32,32):fill(0)

model:cuda()
model:evaluate()

for n=1,test.data:size(1),64 do
  input = test.data[{{n,(n-1)+64},{},{},{}}]
  label = test.labels[{n,(n-1)+64}]
  output = model:forward(input:cuda())
  _, preds = output:max(2)
  preds:eq(label):sum()
end
