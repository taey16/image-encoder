
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
  return image_list, label_list, synset_list
end
