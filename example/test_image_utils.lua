
require 'torch'

local image_utils = paths.dofile('../utils/image_utils.lua')

local img_fname = '../img/cat.jpg'

local img = image_utils.loadImage(img_fname)
save_images(img, 3, './loadImage.png')
print( '===> LoadingImage(img_fname)' )

local loadSize = {3, 384, 384}
local img_resize_crop = image_utils.resize_crop(img, loadSize)
save_images(img_resize_crop, 3, './resize_crop.png')
print( '===> resize_crop(img, loadSize)' )

local img_resize_crop = image_utils.loadImage(img_fname, loadSize)
save_images(img_resize_crop, 3, './loadImage.resize_crop.png')
print( '===> loadImage(img_fname, loadSize)' )

local img_random_flip = image_utils.random_flip(img_resize_crop)
save_images(img_random_flip, 3, './random_flip.png')
print( '===> random_flip(img_resize_crop)' )

sampleSize = {3, 224, 224}
local img_center_crop = image_utils.center_crop(img_random_flip, sampleSize)
save_images(img_center_crop, 3, './center_crop.png')
print( '===> center_crop(img_random_flip)' )

local img_augment = image_utils.augment_image(img_random_flip, loadSize, sampleSize)
save_images(img_augment, 10, './img_augment.png')
print( '===> augment_image(img, loadSize, sampleSize)' )

local kernel_size = 5
local img_lcn = image_utils.local_contrast_norm( img_augment[{1,{},{},{}}], kernel_size )
save_images(img_lcn, 3, './img_lcn.png')
print( '===> local_contrast_norm(img, kernelSize)' )

output_resolution = 224
local img_rst = image_utils.random_RST(img_lcn, output_resolution)
save_images(img_rst, 3, './img_rst.png')
print( '===> random_RST(img, output_resolution)' )

