
paths.dofile('../utils/image_utils.lua')

loader = {}

local testHook = function(path, loadSize, sampleSize, mean, std)
  local input = loadImage(path, loadSize)
  local output= center_crop(input, sampleSize)
  output = mean_std_norm(output, mean, std)
  return output
end
loader.testHook = testHook

local inception7_aug20 = function(filename, loadSize, sampleSize, mean, std)
  local input = loadImage(filename, loadSize, 0)
  local input1= loadImage(filename, loadSize, 1)
  input = mean_std_norm(input, mean, std)
  input1= mean_std_norm(input1,mean, std)
  input = augment_image(input, loadSize, sampleSize )
  input1= augment_image(input1,loadSize, sampleSize )
  local output = torch.FloatTensor(20, sampleSize[1], sampleSize[2], sampleSize[3])
  output[{{1 ,10},{},{},{}}] = input
  output[{{11,20},{},{},{}}] = input1

  return output 
end
loader.inception7_aug20 = inception7_aug20

return loader

