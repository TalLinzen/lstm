--
----	Copyright (c) 2014, Facebook, Inc.
----	All rights reserved.
----
----	This source code is licensed under the Apache 2 license found in the
----	LICENSE file in the root directory of this source tree. 
----
require('cunn')
require('nngraph')
require('xlua')

require('base')
require('options')
require('data')
require('model')

deviceParams = cutorch.getDeviceProperties(1)
cudaComputeCapability = deviceParams.major + deviceParams.minor / 10
print('Cuda compute capability: ' .. cudaComputeCapability)

local options = Options()
params = options:parse(arg)

local state_train, state_valid
local model = {}

local function reset_state(state)
	state.pos = 1
	if model ~= nil and model.start_s ~= nil then
		for d = 1, 2 * params.layers do
			model.start_s[d]:zero()
		end
	end
end

local function forward_pass(state)
	g_replace_table(model.s[0], model.start_s)
	if state.pos + params.seq_length > state.data:size(1) then
		reset_state(state)
	end
	for i = 1, params.seq_length do
		local x = state.data[state.pos]
		local y = state.data[state.pos + 1]
		local s = model.s[i - 1]
		model.err[i], model.s[i] = unpack(model.rnns[i]:forward({x, y, s}))
		state.pos = state.pos + 1
	end
	g_replace_table(model.start_s, model.s[params.seq_length])
	return model.err:mean()
end

local function backward_pass(state)
	model.paramdx:zero()
	model:reset_ds()
	for i = params.seq_length, 1, -1 do
		state.pos = state.pos - 1
		local x = state.data[state.pos]
		local y = state.data[state.pos + 1]
		local s = model.s[i - 1]
		local derr = torch.ones(1):cuda()
		local tmp = model.rnns[i]:backward({x, y, s}, {derr, model.ds})[3]
		g_replace_table(model.ds, tmp)
		cutorch.synchronize()
	end
	state.pos = state.pos + params.seq_length
	model.norm_dw = model.paramdx:norm()
	if model.norm_dw > params.max_grad_norm then
		local shrink_factor = params.max_grad_norm / model.norm_dw
		model.paramdx:mul(shrink_factor)
	end
	model.paramx:add(model.paramdx:mul(-state.learning_rate))
end

local function run_valid()
	reset_state(state_valid)
    model:disable_dropout()
	local len = (state_valid.data:size(1) - 1) / (params.seq_length)
	local perp = 0
	for i = 1, len do
		perp = perp + forward_pass(state_valid)
	end
	print("\nValidation set perplexity: " .. g_f3(torch.exp(perp / len)))
	model:enable_dropout()
end

function run_test(model, corpus)
    local test = corpus.test
    local test = test:resize(test:size(1), 1):expand(test:size(1), params.batch_size)
	local state_test = {data=test:cuda()}
	reset_state(state_test)
    model:disable_dropout()
	local perp = 0
	local len = state_test.data:size(1)
	g_replace_table(model.s[0], model.start_s)
    
	for i = 1, (len - 1) do
		local x = state_test.data[i]
		local y = state_test.data[i + 1]
		perp_tmp, model.s[1] = unpack(model.rnns[1]:forward({x, y, model.s[0]}))
		perp = perp + perp_tmp[1]
		g_replace_table(model.s[0], model.s[1])
	end
	print("Test set perplexity : " .. g_f3(torch.exp(perp / (len - 1))))
    model:enable_dropout()
end

function main()
	g_init_gpu(arg)

    if params.save_dir ~= nil then
       if paths.dirp(params.save_dir) == false then
           os.execute('mkdir -p ' .. params.save_dir)
       end
       print('Models will be saved after each epoch.')
    end

    corpus = Corpus(params.dataset, params.vocab_size)
    corpus:load_datasets()
	state_train = {learning_rate=params.learning_rate}
    local valid = replicate(corpus.valid, params.batch_size)
	state_valid = {data=valid:cuda()}

    model = Model(params.seq_length, params.n_hidden, params.layers,
        params.batch_size, params.vocab_size, params.initial_weight,
        params.dropout)

	print("Network parameters:")
	print(params)
    reset_state(state_train)

	local epoch = 0
	local beginning_time = torch.tic()
	local start_time = torch.tic()
	print("Starting training.")
	local words_per_step = params.seq_length * params.batch_size
	local epoch_size = torch.floor(corpus.train:size(1) / words_per_step)
	local perps
    local chunk_size_in_batches = torch.floor(epoch_size / params.n_chunks)
    local chunk_size_in_data = torch.floor(corpus.train:size(1) / params.n_chunks)

	while epoch < params.n_epochs do
        local step = 0
        local total_cases = 0
	    local epoch_start_time = torch.tic()
        local chunk_index = 0
        if epoch >= params.n_epochs_before_decay then
            state_train.learning_rate = state_train.learning_rate / params.learning_rate_decay
        end
        while chunk_index < params.n_chunks do
            if step % chunk_size_in_batches == 0 then
                local first = chunk_size_in_data * chunk_index + 1
                local last = first + chunk_size_in_data
                -- make last chunk slightly longer to include last few words
                if corpus.train:size(1) - last < params.n_chunks then
                    last = corpus.train:size(1)
                end
                local chunk = corpus.train[{{first, last}}]
                state_train.data = replicate(chunk, params.batch_size):cuda()
                chunk_index = chunk_index + 1
            end
            local perp = forward_pass(state_train)
            if perps == nil then
                perps = torch.zeros(epoch_size):add(perp)
            end
            perps[step + 1] = perp
            step = step + 1
            backward_pass(state_train)
            total_cases = total_cases + params.seq_length * params.batch_size
            xlua.progress(step, epoch_size)

            if step % 33 == 0 then
                cutorch.synchronize()
                collectgarbage()
            end
        end

        -- End-of-epoch bookkeeping
        local wps = torch.floor(total_cases / torch.toc(epoch_start_time))
        local minutes = torch.round(torch.toc(beginning_time) / 60)
        print('epoch = ' .. epoch ..
                    ', training perplexity = ' .. g_f3(torch.exp(perps:mean())) ..
                    ', word per second = ' .. wps ..
                    ', dw:norm() = ' .. g_f3(model.norm_dw) ..
                    ', learning_rate = ' ..  g_f3(params.learning_rate) ..
                    ', time elapsed = ' .. string.format('%d', minutes) .. ' mins.')
        run_valid()
        if params.save_dir ~= nil then
            print 'Saving model to disk...\n'
            local serialized = {}
            serialized.model = model
            serialized.current_learning_rate = state_train.learning_rate
            serialized.params = params
            local filename = 'model_' .. torch.floor(epoch)
            torch.save(paths.concat(params.save_dir, filename), serialized)
        end
        epoch = epoch + 1
	end
    print("Calculating test set perplexity...")
	run_test(model, corpus)
	print("Training is over.")
end

function load_model(epoch)
    local filename = 'model_' .. torch.floor(epoch)
    serialized = torch.load(paths.concat(params.save_dir, filename))
    return(serialized)
end
