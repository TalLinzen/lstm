--
----  Copyright (c) 2014, Facebook, Inc.
----  All rights reserved.
----
----  This source code is licensed under the Apache 2 license found in the
----  LICENSE file in the root directory of this source tree. 
----

local stringx = require('pl.stringx')
local file = require('pl.file')

local data_path = "./data/"

local unk = 1
local vocab_idx = 1
local vocab_map = {}
word_freq = {}

-- Stacks replicated, shifted versions of x_inp
-- into a single matrix of size x_inp:size(1) x batch_size.
function replicate(x_inp, batch_size)
    local s = x_inp:size(1)
    local x = torch.zeros(torch.floor(s / batch_size), batch_size)
    for i = 1, batch_size do
        local start = torch.round((i - 1) * s / batch_size) + 1
        local finish = start + x:size(1) - 1
        x:sub(1, x:size(1), i, i):copy(x_inp:sub(start, finish))
    end
    return x
end

local function load_data(fname, params, fixed_vocab)
    local word_count = 0
    for line in io.lines(fname) do
        local words = stringx.split(line)
        for k = 1, #words do
            if not fixed_vocab then
                if vocab_map[words[k]] == nil then
                    vocab_idx = vocab_idx + 1
                    vocab_map[words[k]] = vocab_idx
                    word_freq[words[k]] = 0
                end
                word_freq[words[k]] = word_freq[words[k]] + 1
            end
            word_count = word_count + 1
        end
    end

    local unk_types = 0
    if not fixed_vocab then
        local freqs = torch.zeros(vocab_idx)
        local i = 1
        for word, freq in pairs(word_freq) do
            freqs[i] = freq
            i = i + 1
        end
        sorted, indices = torch.sort(freqs, 1, true)  -- descending
        threshold = sorted[params.vocab_size] + 1

        local new_vocab_idx = 1
        for word, freq in pairs(word_freq) do
            if freq < threshold then
                vocab_map[word] = unk
                unk_types = unk_types + 1
            else
                new_vocab_idx = new_vocab_idx + 1
                vocab_map[word] = new_vocab_idx
            end
        end
    end

    local x = torch.zeros(word_count)
    local i = 1
    local unk_tokens = 0

    for line in io.lines(fname) do
        local words = stringx.split(line)
        for k = 1, #words do
            if vocab_map[words[k]] == nil then
                x[i] = unk
                unk_tokens = unk_tokens + 1
            else
                x[i] = vocab_map[words[k]]
                if x[i] == unk then
                    unk_tokens = unk_tokens + 1
                end
            end
            i = i + 1
        end
    end

    print('Loaded dataset ' .. fname .. ' (' .. word_count .. ' words)')
    print('Unknown types: ' .. unk_types .. '; unknown tokens: ' .. unk_tokens)
    return x
end

function load_datasets(params)
    local path = paths.concat(data_path, params.dataset)
    res = {}
    res.train = load_data(paths.concat(path, "train.txt"), params, false)
    res.valid = load_data(paths.concat(path, "valid.txt"), params, true)
    res.test = load_data(paths.concat(path, "test.txt"), params, true)
    return res
end
