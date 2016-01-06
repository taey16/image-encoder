
-- Loads the mapping from model outputs to human readable labels
function load_synset()
  local file = io.open '/storage/ImageNet/ILSVRC2012/synset_words.txt'
  local list = {}
  while true do
    local line = file:read()
    if not line then break end
    table.insert(list, string.sub(line,11))
  end
  return list
end


function get_val()
  local file = io.open '/storage/ImageNet/ILSVRC2012/val_synset.txt'
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


function get_test()
  local file = io.open('/storage/ImageNet/ILSVRC2012/test.txt', 'r')
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
      string.format(
        'http://10.202.4.219:2596/PBrain/ImageNet/ILSVRC2012/test/%s', 
      item[1])
  end
  file:close()
  return image_list, label_list, list_table
end


