--
----  Copyright (c) 2014, Facebook, Inc.
----  All rights reserved.
----
----  This source code is licensed under the Apache 2 license found in the
----  LICENSE file in the root directory of this source tree. 
----

local stringx = require('pl.stringx')
local file = require('pl.file')

local ptb_path = "./data/"

local vocab_idx = 0
local vocab_map = {}

-- Stacks replicated, shifted versions of x_inp
-- into a single matrix of size x_inp:size(1) x batch_size.
local function replicate(x_inp, batch_size)
   local s = x_inp:size(1)
   local x = torch.zeros(torch.floor(s / batch_size), batch_size)
   for i = 1, batch_size do
     local start = torch.round((i - 1) * s / batch_size) + 1
     local finish = start + x:size(1) - 1
     x:sub(1, x:size(1), i, i):copy(x_inp:sub(start, finish))
   end
   return x
end

local function load_data(fname)
   local data = file.read(fname)
   data = stringx.replace(data, '\n', '<eos>')
   data = stringx.split(data)
   print(string.format("Loading %s, size of data = %d", fname, #data))
   local x = torch.zeros(#data)
   for i = 1, #data do
      if vocab_map[data[i]] == nil then
         vocab_idx = vocab_idx + 1
         vocab_map[data[i]] = vocab_idx
      end
      x[i] = vocab_map[data[i]]
   end
   return x
end

function load_datasets(params)
    local path = paths.concat('./data/', params.dataset)
    local train = load_data(paths.concat(path, "train.txt"))
    local valid = load_data(paths.concat(path, "valid.txt"))
    local test = load_data(paths.concat(path, "test.txt"))

    res = {}
    res.train = replicate(train, params.batch_size)
    res.valid = replicate(valid, params.batch_size)

    -- Original comment from Zaremba:
    -- Intentionally we repeat dimensions without offseting.
    -- Pass over this batch corresponds to the fully sequential processing.
    res.test = test:resize(test:size(1), 1):expand(test:size(1), params.batch_size)

    return res
end
