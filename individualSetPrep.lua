--
-- Created by IntelliJ IDEA.
-- User: james
-- Date: 05/05/16
-- Time: 8:57 AM
-- To change this template use File | Settings | File Templates.
--

require 'torch'
require 'image'
require 'os'
local pl = require('pl.import_into')()
local magick = require('magick')


local input_sub = 128
-- Scale input image
local input_scale = 0.0078125
-- input dimension
local input_dim = 250

function load_image(path)
    local img   = image.load(path, 3)
    local w, h  = img:size(3), img:size(2)
    local min   = math.min(w, h)
    img         = image.crop(img, 'c', min, min)
    img         = image.scale(img, input_dim)
    -- normalize image
    img:mul(255):add(-input_sub):mul(input_scale)
    -- due to batch normalization we must use minibatches
    return img:float():view(1, img:size(1), img:size(2), img:size(3))
end
function file_exists(name)
    local f=io.open(name,"r")
    if f~=nil then io.close(f) return true else return false end
end


function catTables(tableA, tableB)
    local tableC = {}
    for i=1, #tableA do
        tableC[i] = tableA[i]
    end
    for i=1, #tableB do
        tableC[i+#tableA] = tableB[i]
    end
    return tableC
end


local set = ...

--local net = torch.load(args.m)
--net:evaluate()

local new_set = {data, label}
new_set.data = {}
new_set.label = {}
local first_pass_jpg = pl.dir.getallfiles('/home/james/reddit/' .. set, '*.jpg')
local first_pass_png = pl.dir.getallfiles('/home/james/reddit/' .. set, '*.png')
local first_pass = catTables(first_pass_jpg, first_pass_png)
local class
print("removing bad files.")
--deletes files that are invalid and unloadable by imagemagick.
for i, f in ipairs(first_pass) do
    local checker, img = pcall(image.load, f)
    if not(checker) then
        print("removed file: " .. f)
        os.remove(f)
    end
    collectgarbage()
end
print("loading images into tensors")
local second_pass_jpg = pl.dir.getallfiles('/home/james/reddit/' .. set, '*.jpg')
local second_pass_png = pl.dir.getallfiles('/home/james/reddit/' .. set, '*.png')
local second_pass = catTables(second_pass_png, second_pass_jpg)
for i, f in ipairs(second_pass) do
    new_set.data[i] = load_image(f);
    new_set.label[i] = set;
    collectgarbage()
end

print("got all images loaded and resized.")


if(file_exists('/home/james/models/subredditData/completeset.dat')) then
    local completeset = torch.load('/home/james/models/subredditData/completeset.dat')
    local tempData = torch.FloatTensor(completeset.data:size(1) + #new_set.data, 3, input_dim, input_dim)
    local tempLabel = torch.LongTensor(completeset.label:size(1) + #new_set.label)
    collectgarbage()
    for i=1, completeset.data:size(1) do
        tempData[i] = completeset.data[i]
        tempLabel[i] = completeset.label[i]
    end
    class = completeset.label[#completeset.label]+1
    local start = completeset.label:size(1)

    for i=1, #new_set.data do
        tempData[i + start] = new_set.data[i]
        tempLabel[i + start] = class
    end
    completeset.data = tempData
    completeset.label = tempLabel
    print(set, class)
    torch.save('/home/james/models/subredditData/completeset.dat', completeset)

else
    local completeset =  {data, label }
    class = 1
    completeset.data = torch.FloatTensor(#new_set.data, 3, input_dim, input_dim)
    completeset.label = torch.LongTensor(#new_set.label)
    for i=1, #new_set.data do
        completeset.data[i] = new_set.data[i]
        completeset.label[i] = class
        end
    print(set, class)
    torch.save('/home/james/models/subredditData/completeset.dat', completeset)
end
