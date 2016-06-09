
package.path = '../?.lua;'..package.path
local image_utils = require 'utils.image_utils'

loader = {}

local testHook = function(path, loadSize, sampleSize, mean, std)
  local input = image_utils.loadImage(path, loadSize)
  local output= image_utils.center_crop(input, sampleSize)
  output = image_utils.mean_std_norm(output, mean, std)
  return output
end
loader.testHook = testHook

local inception_v3_aug20 = function(filename, loadSize, sampleSize, mean, std)
  local input = image_utils.loadImage(filename, loadSize, 0)
  local input1= image_utils.loadImage(filename, loadSize, 1)
  input = image_utils.mean_std_norm(input, mean, std)
  input1= image_utils.mean_std_norm(input1,mean, std)
  input = image_utils.augment_image(input, loadSize, sampleSize )
  input1= image_utils.augment_image(input1,loadSize, sampleSize )
  local output = torch.FloatTensor(20, sampleSize[1], sampleSize[2], sampleSize[3])
  output[{{1 ,10},{},{},{}}] = input
  output[{{11,20},{},{},{}}] = input1

  return output 
end
loader.inception_v3_aug20 = inception_v3_aug20


return loader

