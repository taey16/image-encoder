
local attribute_utils = {}


-- Loads the mapping from model outputs to human readable labels
function attribute_utils.get_val(attribute_id)
  local filename = string.format(
    '/storage/freebee/tshirts_shirts_blous_knit.image_sentence.txt.json.%s.val.txt', attribute_id)
  local file = io.open(filename)
  if file == nil then
    error(string.format(
      'attribute_utils.get_val, check filename: %s', filename))
  end
  local image_list = {}
  local label_list = {}
  while true do
    local line = file:read()
    if not line then break end
    local item = string.split(line, ' ')
    table.insert(image_list, item[1])
    table.insert(label_list, 2-tonumber(item[2]))
  end
  file.close()
  return image_list, label_list
end

return attribute_utils
