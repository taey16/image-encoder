
local imagenet_utils = {}

-- Loads the mapping from model outputs to human readable labels
function imagenet_utils.load_synset(_filename)
  filename = _filename or '/storage/ImageNet/ILSVRC2012/synset_words.txt'
  local file = io.open(filename, 'r')
  local list = {}
  while true do
    local line = file:read()
    if not line then break end
    table.insert(list, string.sub(line,11))
  end
  file:close()
  return list
end


function imagenet_utils.get_val(_filename)
  local filename = _filename or '/storage/ImageNet/ILSVRC2012/val_synset.txt'
  local file = io.open(filename, 'r')
  local image_list ={}
  local label_list ={}
  local synset_list={}

  while true do
    local line = file:read()
    if not line then break end
    local item  =string.split(line, ' ')
    local synset=string.split(item[1], '/')
    table.insert(image_list, item[1])
    table.insert(label_list, item[2])
    table.insert(synset_list,synset[1])
  end
  file:close()
  return image_list, label_list, synset_list
end


function imagenet_utils.get_test(_filename)
  local filename = _filename or '/storage/ImageNet/ILSVRC2012/test.txt'
  local file = io.open(filename, 'r')
  local image_list = {}
  local label_list = {}
  local list_table = {}
  while true do
    local line = file:read()
    if not line then break end
    local item = string.split(line, ' ')
    table.insert(image_list, item[1])
    table.insert(label_list, item[2])
    list_table[item[1]] = 
      string.format('http://10.202.4.219:2596/PBrain/ImageNet/ILSVRC2012/test/%s', item[1])
  end
  file:close()
  return image_list, label_list, list_table
end

return imagenet_utils


