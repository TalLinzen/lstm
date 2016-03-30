--
----  Copyright (c) 2014, Facebook, Inc.
----  All rights reserved.
----
----  This source code is licensed under the Apache 2 license found in the
----  LICENSE file in the root directory of this source tree. 
----

local stringx = require('pl.stringx')
local file = require('pl.file')

local Corpus = torch.class('Corpus')

local data_path = "./data/"
local unk = 1

function Corpus:__init(dataset, vocab_size)
    self.path = paths.concat(data_path, dataset)
    self.vocab_size = vocab_size
    self.vocab_idx = 1
    self.vocab_map = {}
    self.word_freq = {}
end

function Corpus:encode_line(line, x, i)
    local words = stringx.split(line)
    local unk_tokens = 0

    for k = 1, #words do
        if self.vocab_map[words[k]] == nil then
            x[i] = unk
            unk_tokens = unk_tokens + 1
        else
            x[i] = self.vocab_map[words[k]]
            if x[i] == unk then
                unk_tokens = unk_tokens + 1
            end
        end
        i = i + 1
    end

    return unk_tokens, #words
end

function Corpus:load_data(fname, fixed_vocab)
    local word_count = 0
    for line in io.lines(fname) do
        local words = stringx.split(line)
        for k = 1, #words do
            if not fixed_vocab then
                if self.vocab_map[words[k]] == nil then
                    self.vocab_idx = self.vocab_idx + 1
                    self.vocab_map[words[k]] = self.vocab_idx
                    self.word_freq[words[k]] = 0
                end
                self.word_freq[words[k]] = self.word_freq[words[k]] + 1
            end
            word_count = word_count + 1
        end
    end

    local unk_types = 0
    if not fixed_vocab then
        local freqs = torch.zeros(self.vocab_idx)
        local i = 1
        for word, freq in pairs(self.word_freq) do
            freqs[i] = freq
            i = i + 1
        end
        sorted, indices = torch.sort(freqs, 1, true)  -- descending
        threshold = sorted[self.vocab_size] + 1

        local new_vocab_idx = 1
        for word, freq in pairs(self.word_freq) do
            if freq < threshold then
                self.vocab_map[word] = unk
                unk_types = unk_types + 1
            else
                new_vocab_idx = new_vocab_idx + 1
                self.vocab_map[word] = new_vocab_idx
            end
        end
    end

    local x = torch.zeros(word_count)
    local i = 1
    local unk_tokens = 0

    for line in io.lines(fname) do
        ut, nwords = self:encode_line(line, x, i)
        i = i + nwords
        unk_tokens = unk_tokens + ut
    end

    print('Loaded dataset ' .. fname .. ' (' .. word_count .. ' words)')
    print('Unknown types: ' .. unk_types .. '; unknown tokens: ' .. unk_tokens)
    return x
end

function Corpus:load_datasets()
    self.train = self:load_data(paths.concat(self.path, "train.txt"), false)
    self.valid = self:load_data(paths.concat(self.path, "valid.txt"), true)
    self.test = self:load_data(paths.concat(self.path, "test.txt"), true)
end

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
