
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

  while true do
    local line = file:read()
    if not line then break end
    item = string.split(line, ' ')
    table.insert(image_list, item[1])
    table.insert(label_list, item[2])
  end
  return image_list, label_list
end
