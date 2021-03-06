
require 'torch'
torch.setdefaulttensortype('torch.FloatTensor')
local ffi = require 'ffi'
local class = require('pl.class')
local dir = require 'pl.dir'
local tablex = require 'pl.tablex'
local argcheck = require 'argcheck'
require 'sys'
require 'xlua'

local dataset = torch.class('dataLoader')

local initcheck = argcheck{
  pack=true,
  help=[[
    A dataset class for images in a flat folder structure (folder-name is class-name).
    Optimized for extremely large datasets (upwards of 14 million images).
    Tested only on Linux (as it uses command-line linux utilities to scale up)
  ]],
  {check=function(paths)
     local out = true;
       for k,v in ipairs(paths) do
         if type(v) ~= 'string' then
           print('paths can only be of string input');
           out = false
         end
       end
       return out
   end,
   name="paths", type="table", help="Multiple paths of directories with images"}, 
  {name="sampleSize", type="table", help="a consistent sample size to resize the images"},
  {name="verbose", type="boolean", help="Verbose mode during initialization", default = false},
  {name="loadSize", type="table", help="a size to load the images to, initially", opt = true},
  {name="forceClasses",
     type="table",
     help="If you want this loader to map certain classes to certain indices, "
       .. "pass a classes table that has {classname : classindex} pairs."
       .. " For example: {3 : 'dog', 5 : 'cat'}"
       .. "This function is very useful when you want two loaders to have the same "
       .. "class indices (trainLoader/testLoader for example)", opt = true},
  {name="sampleHookTrain",
     type="function",
     help="applied to sample during training(ex: for lighting jitter). "
          .."It takes the image path as input", opt = true},
  {name="sampleHookTest", type="function", help="applied to sample during testing", opt = true},
}

function dataset:__init(...)
  -- argcheck
  local args =  initcheck(...)
  for k,v in pairs(args) do self[k] = v end

  if not self.loadSize then 
    self.loadSize = self.sampleSize; 
  end
  if not self.sampleHookTrain then 
    self.sampleHookTrain = self.defaultSampleHook
  end
  if not self.sampleHookTest then 
    self.sampleHookTest = self.defaultSampleHook
  end

  -- find class names
  self.classes = {}
  local classPaths = {}
  if self.forceClasses then
    for k,v in pairs(self.forceClasses) do
      self.classes[k] = v
      classPaths[k] = {}
    end
  end

  local function tableFind(t, o) 
    for k, v in pairs(t) do 
      if v == o then return k end 
    end 
  end

  -- loop over each paths folder, get list of unique class names,
  -- also store the directory paths per class
  -- for each class,
  for k, path in ipairs(self.paths) do
    local dirs = dir.getdirectories(path);
    for k,dirpath in ipairs(dirs) do
      local class = paths.basename(dirpath)
      local idx = tableFind(self.classes, class)
      if not idx then
        table.insert(self.classes, class)
        idx = #self.classes
        classPaths[idx] = {}
      end
      if not tableFind(classPaths[idx], dirpath) then
        table.insert(classPaths[idx], dirpath);
      end
    end
  end

  self.classIndices = {}
  for k,v in ipairs(self.classes) do
    self.classIndices[v] = k
  end

  local wc  = 'wc'
  local cut = 'cut'
  local find= 'find'

  -- Options for the GNU find command
  local extensionList = {'jpg', 'png','JPG','PNG','JPEG', 'ppm', 'PPM', 'bmp', 'BMP'}
  local findOptions = ' -iname "*.' .. extensionList[1] .. '"'
  for i=2,#extensionList do
    findOptions = findOptions .. ' -o -iname "*.' .. extensionList[i] .. '"'
  end

  -- find the image path names
  -- path to each image in dataset
  self.imagePath = torch.CharTensor()
  -- class index of each image (class index in self.classes)
  self.imageClass= torch.LongTensor()
  -- index of imageList to each image of a particular class
  self.classList = {}
  -- the main list used when sampling data
  self.classListSample = self.classList

  print(
    'running "find" on each class directory, and concatenate all'
    .. ' those filenames into a single file containing all image paths for a given class')
  -- so, generates one file per class
  local classFindFiles = {}
  for i=1,#self.classes do
    classFindFiles[i] = os.tmpname()
  end
  local combinedFindList = os.tmpname();

  local tmpfile = os.tmpname()
  local tmphandle = assert(io.open(tmpfile, 'w'))
  -- iterate over classes
  for i, class in ipairs(self.classes) do
    -- iterate over classPaths
    for j,path in ipairs(classPaths[i]) do
      local command = find .. ' "' .. path .. '" ' .. findOptions
          .. ' >>"' .. classFindFiles[i] .. '" \n'
      tmphandle:write(command)
    end
  end
  io.close(tmphandle)
  os.execute('bash ' .. tmpfile)
  os.execute('rm -f ' .. tmpfile)

  print('now combine all the files to a single large file')
  local tmpfile = os.tmpname()
  local tmphandle = assert(io.open(tmpfile, 'w'))
  -- concat all finds to a single large file in the order of self.classes
  for i=1,#self.classes do
    local command = 'cat "' .. classFindFiles[i] .. '" >>' .. combinedFindList .. ' \n'
    tmphandle:write(command)
  end
  io.close(tmphandle)
  os.execute('bash ' .. tmpfile)
  os.execute('rm -f ' .. tmpfile)

  print('load the large concatenated list of sample paths to self.imagePath')
  local maxPathLength = 
    tonumber(sys.fexecute(wc .. " -L '" .. combinedFindList .. "' |" .. cut .. " -f1 -d' '")) + 1
  local length = 
    tonumber(sys.fexecute(wc .. " -l '" .. combinedFindList .. "' |" .. cut .. " -f1 -d' '"))

  assert(length > 0, "Could not find any image file in the given input paths")
  assert(maxPathLength > 0, "paths of files are length 0?")
  self.imagePath:resize(length, maxPathLength):fill(0)
  local s_data = self.imagePath:data()
  local count = 0
  for line in io.lines(combinedFindList) do
    ffi.copy(s_data, line)
    s_data = s_data + maxPathLength
    if self.verbose and count % 10000 == 0 then
      xlua.progress(count, length)
    end
    count = count + 1
  end

  self.numSamples = self.imagePath:size(1)
  if self.verbose then print(self.numSamples ..  ' samples found.') end
  print('Updating classList and imageClass appropriately')
  self.imageClass:resize(self.numSamples)
  local runningIndex = 0
  for i=1,#self.classes do
    if self.verbose then xlua.progress(i, #(self.classes)) end
    local length = tonumber(sys.fexecute(wc .. " -l '"
                                            .. classFindFiles[i] .. "' |"
                                            .. cut .. " -f1 -d' '"))
    if length == 0 then
      error('Class has zero samples')
    else
      self.classList[i] = torch.linspace(runningIndex + 1, runningIndex + length, length):long()
      self.imageClass[{{runningIndex + 1, runningIndex + length}}]:fill(i)
    end
    runningIndex = runningIndex + length
  end

  -- clean up temporary files
  print('Cleaning up temporary files')
  local tmpfilelistall = ''
  for i=1,#(classFindFiles) do
    tmpfilelistall = tmpfilelistall .. ' "' .. classFindFiles[i] .. '"'
    if i % 1000 == 0 then
      os.execute('rm -f ' .. tmpfilelistall)
      tmpfilelistall = ''
    end
  end
  os.execute('rm -f '  .. tmpfilelistall)
  os.execute('rm -f "' .. combinedFindList .. '"')

  self.testIndicesSize = 0
end


-- class:= {'class_name_1', 'class_name_2', ...}
function dataset:size(class, list)
  list = list or self.classList
  if not class then
    return self.numSamples
  elseif type(class) == 'string' then
    return list[self.classIndices[class]]:size(1)
  elseif type(class) == 'number' then
    return list[class]:size(1)
  end
end


-- converts a table of samples (and corresponding labels) to a clean tensor
local function tableToOutput(self, dataTable, scalarTable)
  local data, scalarLabels, labels
  local quantity = #scalarTable
  assert(dataTable[1]:dim() == 3)
  data = torch.Tensor(
    quantity, self.sampleSize[1], self.sampleSize[2], self.sampleSize[3])
  scalarLabels = torch.LongTensor(quantity):fill(-1111)
  for i=1,#dataTable do
    data[i]:copy(dataTable[i])
    scalarLabels[i] = scalarTable[i]
  end
  return data, scalarLabels
end


function dataset:getByClass(class)
  local index = 
    math.ceil(torch.uniform() * self.classListSample[class]:nElement())
  local imgpath = 
    ffi.string(torch.data(self.imagePath[self.classListSample[class][index]]))
  return self:sampleHookTrain(imgpath)
end


function dataset:stratified_sample(quantity)
  local ratio = self:size('button') / self:size('non_button')
  local dataTable = {}
  local scalarTable = {}
  for i=1,quantity do
    local _rnd = torch.uniform()
    local class
    if ratio < _rnd then class = 1
    else class = 2 end
    local out = self:getByClass(class)
    table.insert(dataTable, out)
    table.insert(scalarTable, class)
  end
  local data, scalarLabels = tableToOutput(self, dataTable, scalarTable)
  return data, scalarLabels
end


function dataset:sample(quantity)
  --assert(quantity > 0)
  local dataTable = {}
  local scalarTable = {}
  for i=1,quantity do
    local class = torch.random(1, #self.classes)
    local out = self:getByClass(class)
    table.insert(dataTable, out)
    table.insert(scalarTable, class)
  end
  local data, scalarLabels = tableToOutput(self, dataTable, scalarTable)
  return data, scalarLabels
end


function dataset:get(i1, i2)
  local indices = torch.range(i1, i2);
  local quantity = i2 - i1 + 1;
  local dataTable = {}
  local scalarTable = {}
  for i=1,quantity do
    -- load the sample
    local imgpath = ffi.string(torch.data(self.imagePath[indices[i]]))
    local out = self:sampleHookTest(imgpath)
    table.insert(dataTable, out)
    table.insert(scalarTable, self.imageClass[indices[i]])
  end
  local data, scalarLabels = tableToOutput(self, dataTable, scalarTable)
  return data, scalarLabels
end

return dataset

