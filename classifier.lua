--
-- Created by IntelliJ IDEA.
-- User: james
-- Date: 05/05/16
-- Time: 8:35 AM
-- To change this template use File | Settings | File Templates.
--
require 'torch'
require 'image'
require 'nn'
require 'cunn'
local pl = require('pl.import_into')()


--gets the top 3 classes for the provided image tensor.
function guess(model, imageTensor, classes)
    local prediction = model:forward(imageTensor):double()
    for i=1, prediction:size(1) do
        prediction[i] = math.exp(prediction[i])
    end
    local confidence, ind = torch.sort(prediction, true)
    local result = {}
    for i=1, 3 do
        result[i] = classes[ind[i]] .. ',' .. confidence[i]
    end
    return result
end

--calculates the average top 1 -> top 3 class accuracies for the provided dataset.
function accuracy(model, dataset)
    local top1 = 0
    local top2 = 0
    local top3 = 0
    for i=1,dataset:size() do
        local groundtruth = dataset.label[i]
        local prediction = model:forward(dataset.data[i])
        local confidences, indices = torch.sort(prediction, true)  -- true means sort in descending order
        if groundtruth == indices[1] then
            top1 = top1 + 1
            top2 = top2 + 1
            top3 = top3 + 1
        elseif groundtruth == indices[2] then
            top2 = top2 + 1
            top3 = top3 + 1
        elseif groundtruth == indices[3] then
            top3 = top3 + 1
        end
    end
    print(top1, " top 1 " .. 100*top1/dataset:size() .. ' % ')
    print(top2, " top 2 " .. 100*top2/dataset:size() .. ' % ')
    print(top3, " top 3 " .. 100*top3/dataset:size() .. ' % ')
end


function classAccuracy(model, dataset, classes)
local class_performance = {}
local class_number = {}
for i=1, #classes do
    class_performance[i] = 0
    class_number[i] = 0
end
for i=1, dataset:size() do
    local groundtruth = dataset.label[i]
    class_number[groundtruth] = class_number[groundtruth] + 1
    local prediction = model:forward(dataset.data[i])
    local confidences, indices = torch.sort(prediction, true)  -- true means sort in descending order
    if groundtruth == indices[1] then
        class_performance[groundtruth] = class_performance[groundtruth] + 1
    end
end
for i=1, #classes do
    print(classes[i], class_number[i], 100*class_performance[i]/class_number[i] .. '%')
end
end

local input_sub = 128
-- Scale input image
local input_scale = 0.0078125
-- input dimension
local input_dim = 250

local load_image = function(path)
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


local args = pl.lapp [[
  -m (string) classification model file
  -c (string) class definition file
  -i (string) image file
]]

local model = torch.load(args.m):cuda()
model:evaluate()
local image = load_image(args.i):cuda()

local classes = torch.load(args.c)

local output = guess(model, image, classes)

--print out the top 3 class pairs to stdout
for _, l in ipairs(output) do io.write(l, "\n") end