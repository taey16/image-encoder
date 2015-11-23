require 'torch'
require 'optim'

require 'rosenbrock'
require 'l2'

x = torch.Tensor(2):fill(0)
fx = {}

max_iter = 10001
config = {learningRate=1e-3, momentum=0.9, dampening=0}
for i = 1,max_iter do
	--x, f = optim.sgd(rosenbrock, x, config)
	x, f = optim.nag(rosenbrock, x, config)
	if (i-1)%100 == 0 then
		table.insert(fx,f[1])
	end
end

print()
print('Rosenbrock test')
print()
print('x=');print(x)
print('fx=')
for i=1,#fx do 
  print((i-1)*100+1,fx[i]); 
end
