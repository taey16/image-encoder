
require 'torch'
require 'image'
require 'nn'

function local_contrast_norm( rgb, kernel_size )
  local rgb = rgb:double()
  local kernel_size = kernel_size or 7
  local normalization = 
    nn.SpatialContrastiveNormalization(1, image.gaussian1D(kernel_size))
  local yuv = image.rgb2yuv(rgb)
  -- normalize y locally:
  yuv[1] = normalization(yuv[{{1}}])
  -- normalize u/v globally:
  for c=2,3 do 
    local mean= yuv[c]:mean()
    local std = yuv[c]:std()
    yuv[c]:add(-mean)
    yuv[c]:div(std)
  end
  return image.yuv2rgb(yuv):float()
end


-- use in vgg16Caffe
function preprocess(im)
  local input = image.scale(im,256,256,'bilinear')*255
  if input:dim() == 2 then
    input = input:view(1,input:size(1), input:size(2)):repeatTensor(3,1,1)
  elseif input:dim() == 3 and input:size(1) == 1 then
    input = input:repeatTensor(3,1,1)
  elseif input:dim() == 3 and input:size(1) == 3 then
  elseif input:dim() == 3 and input:size(1) == 4 then
    input = input[{{1,3},{},{}}]
  else
    print(#input)
    error('not 2-channel or 3-channel image')
  end
  -- RGB2BGR
  local output = input:clone()
  output[{1,{},{}}] = input[{3,{},{}}]
  output[{3,{},{}}] = input[{1,{},{}}]
  -- subtract imagemodel mean for vgg16 model
  output[{{1},{},{}}]:add(-103.939)
  output[{{2},{},{}}]:add(-116.779)
  output[{{3},{},{}}]:add(-123.68)

  return output
end


function save_images(x, n, file, padding, nrow, symmetric)
  local padding = padding or 2
  local nrow = nrow or 8
  local symmetric = symmetric or true
  local file = file or "./save_images.png"
  local input = x:narrow(1, 1, n)
  local view = image.toDisplayTensor(
    {input = input, padding = padding, nrow = nrow, symmetric = symmetric}
  )
  image.save(file, view)
end


function augment_image(input, loadSize, sampleSize)
  local oH = sampleSize[2]
  local oW = sampleSize[3]
  local iH = loadSize[2]
  local iW = loadSize[3]
  local w1 = math.ceil((iW-oW)/2)
  local h1 = math.ceil((iH-oH)/2)
  local output = torch.FloatTensor(10, 3, oW, oH)
  output[{1 ,{},{},{}}] = image.crop(input, 1,    1,    oW+1, oH+1)
  output[{2 ,{},{},{}}] = image.crop(input, iW-oW,1,    iH,   oH+1)
  output[{3 ,{},{},{}}] = image.crop(input, 1,    iH-oH,oW+1, iH)
  output[{4 ,{},{},{}}] = image.crop(input, iW-oW,iH-oH,iW,   iH)
  output[{5 ,{},{},{}}] = image.crop(input, w1,   h1,   w1+oW,h1+oH)
  output[{6 ,{},{},{}}] = image.hflip(output[{1,{},{},{}}])
  output[{7 ,{},{},{}}] = image.hflip(output[{2,{},{},{}}])
  output[{8 ,{},{},{}}] = image.hflip(output[{3,{},{},{}}])
  output[{9 ,{},{},{}}] = image.hflip(output[{4,{},{},{}}])
  output[{10,{},{},{}}] = image.hflip(output[{5,{},{},{}}])
  -- save_images(output, 10)
  return output
end


function resize_crop(input, loadSize, preserve_aspect_ratio)
  local output = torch.FloatTensor()
  local preserve_aspect_ratio = 
    preserve_aspect_ratio or torch.uniform()
  if preserve_aspect_ratio > 0.5 then
    local iW = input:size(3)
    local iH = input:size(2)
    if iW < iH then
      output = image.scale(input, 
        loadSize[2], loadSize[2] * iH / iW)
    else
      output = image.scale(input, 
        loadSize[3] * iW / iH, loadSize[3])
    end
  else
    output = image.scale(input, loadSize[2], loadSize[3])
  end
  return output
end


function loadImage(path, loadSize)
  local loadSize = loadSize or nil
  local input = image.load(path)
  if input:dim() == 2 then
    input = input:view(1,input:size(1), input:size(2)):repeatTensor(3,1,1)
  elseif input:dim() == 3 and input:size(1) == 1 then
    input = input:repeatTensor(3,1,1)
  elseif input:dim() == 3 and input:size(1) == 3 then 
  elseif input:dim() == 3 and input:size(1) == 4 then 
    input = input[{{1,3},{},{}}]
  else
    print(#input)
    error('loadImage: not 2-channel or 3-channel image')
  end

  if loadSize then
    input = resize_crop(input, loadSize)
  end

  return input
end


function loadImage(path, loadSize, aspect_ratio)
  local loadSize = loadSize or nil
  local input = image.load(path)
  if input:dim() == 2 then
    input = input:view(1,input:size(1), input:size(2)):repeatTensor(3,1,1)
  elseif input:dim() == 3 and input:size(1) == 1 then
    input = input:repeatTensor(3,1,1)
  elseif input:dim() == 3 and input:size(1) == 3 then 
  elseif input:dim() == 3 and input:size(1) == 4 then 
    input = input[{{1,3},{},{}}]
  else
    print(#input)
    error('loadImage: not 2-channel or 3-channel image')
  end
  input = resize_crop(input, loadSize, aspect_ratio)

  return input
end


function random_RST(img_data, output_resolution)
  local resolution = img_data:size(2)
  local nChannels = img_data:size(1)
  local rotationFactor = 4
  local scaleFactorFrom = 0.7
  local scaleFactorTo = 1.3
  local maximumScaledImgSize = 384

  assert(maximumScaledImgSize > resolution * scaleFactorTo)

  local res=torch.FloatTensor(nChannels, output_resolution, output_resolution):fill(0)
  local distImg=torch.FloatTensor(nChannels, maximumScaledImgSize, maximumScaledImgSize):fill(0)
  baseImg = img_data

  R = image.rotate(baseImg, torch.uniform(-3.14/rotationFactor,3.14/rotationFactor))
  scale = torch.uniform(scaleFactorFrom, scaleFactorTo)
  sz = torch.floor(scale*resolution)
  S = image.scale(R, sz, sz)
  rest = maximumScaledImgSize - sz
  offsetx = torch.random(1, 1+rest)
  offsety = torch.random(1, 1+rest)

  distImg:zero()
  distImg:narrow(2, offsety, sz):narrow(3, offsetx, sz):copy(S)
  res:copy(image.scale(distImg,output_resolution, output_resolution))
  return res
end


function random_flip(input, do_flip)
  local do_flip = do_flip or torch.uniform()
  if do_flip > 0.5 then
    input = image.hflip(input)
  end
  return input
end


function random_jitter(input, sampleSize)
  local iW = input:size(3)
  local iH = input:size(2)
  local oW = sampleSize[3]
  local oH = sampleSize[2]
  local h1 = math.ceil(torch.uniform(1e-2, iH-oH))
  local w1 = math.ceil(torch.uniform(1e-2, iW-oW))
  local output = image.crop(input, w1, h1, w1 + oW, h1 + oH)
  assert(output:size(3) == oW)
  assert(output:size(2) == oH)

  output = random_flip(output)

  return output
end

function mean_std_norm(input, mean, std)
  for i=1,3 do
    if mean then input[{{i},{},{}}]:add(-mean[i]) end
    if  std then input[{{i},{},{}}]:div(std[i]) end
  end
  return input 
end


function center_crop(input, sampleSize)
  local oH = sampleSize[2]
  local oW = sampleSize[3]
  local iW = input:size(3)
  local iH = input:size(2)
  local w1 = math.ceil((iW-oW)/2)
  local h1 = math.ceil((iH-oH)/2)
  local output = image.crop(input, w1, h1, w1+oW, h1+oW)
  return output
end


