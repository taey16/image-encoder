
local fashion_pair_utils = {}


-- Loads the mapping from model outputs to human readable labels
function fashion_pair_utils.get_val(attribute_id)
  local filename = string.format(
    '/storage/freebee/tshirts_shirts_blous_knit.image_sentence.txt.json.%s.val.txt', attribute_id)
  local file = io.open(filename)
  if file == nil then
    error(string.format(
      'fashion_pair_utils.get_val, check filename: %s', filename))
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


function fashion_pair_utils.get_test()
  local filename = string.format(
    '/storage/product/fashion_pair/fashion_pair_test.csv')
  local prefix = '/data1/october_11st/october_11st_imgs/'
  local file = io.open(filename)
  if file == nil then
    error(string.format(
      'fashion_pair_utils.get_test, check filename: %s', filename))
  end
  local image_list_q  = {}
  local image_list_ref= {}
  local label_list = {}
  while true do
    local line = file:read()
    if not line then break end
    local item = string.split(line, ',')
    table.insert(image_list_q,  string.format('%s/%s', prefix, item[1]))
    table.insert(image_list_ref,string.format('%s/%s', prefix, item[2]))
    table.insert(label_list, tonumber(item[3]))
  end
  file.close()
  return image_list_q, image_list_ref, label_list
end

return fashion_pair_utils
