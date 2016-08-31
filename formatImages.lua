--
-- Created by IntelliJ IDEA.
-- User: james
-- Date: 06/05/16
-- Time: 12:05 PM
-- To change this template use File | Settings | File Templates.
--

require 'torch'
require 'math'
require 'cunn'
require 'concurrent'
local pl = require('pl.import_into')()

local args = pl.lapp [[
  -i (string) image dat file
  -j (string) train set output
  -k (string) test set output
  -p (string) ratio of train/test
]]
local imageDims = 250
print(args.j .. " train directory")
print(args.k .. " test directory")
local dataset = torch.load(args.i)

function dataset:size()
    return self.data:size(1)
end

local train_amount = math.floor(dataset:size()*(tonumber(args.p)))
local test_amount = math.floor(dataset:size()*(1-tonumber(args.p)))

function dataset:size()
    return self.data:size(1)
end


function shuffle_dataset(rawset)
local dataSize = rawset:size()
local shuffleIdx = torch.randperm(dataSize)
rawset.data = rawset.data:index(1,shuffleIdx:long())
rawset.label = rawset.label:index(1, shuffleIdx:long())
return rawset
end
--    time to split the dataset


dataset = shuffle_dataset(dataset)

print("shuffled dataset.")
collectgarbage()
local trainset = {}
trainset.data = torch.FloatTensor(train_amount, 3, imageDims, imageDims)
trainset.label = torch.IntTensor(train_amount)

function trainset:size()
return self.data:size(1)
end

print("creating training set..")
for i=1, train_amount do
    trainset.data[i] = dataset.data[i]
    trainset.label[i] = dataset.label[i]
    print(i)
end
print("train set created.")
collectgarbage()
local testset = {}
testset.data = torch.FloatTensor(test_amount, 3, imageDims, imageDims)
testset.label = torch.IntTensor(test_amount)

function testset:size()
return self.data:size(1)
end

collectgarbage()
print("creating test set..")
for i=1, test_amount do
    testset.data[i] = dataset.data[i+train_amount]
    testset.label[i] = dataset.label[i+train_amount]
    print(i)
end
print("test set created.")
collectgarbage()
torch.save(args.j, trainset)
torch.save(args.k, testset)
print("complete")