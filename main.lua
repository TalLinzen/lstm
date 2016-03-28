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

deviceParams = cutorch.getDeviceProperties(1)
cudaComputeCapability = deviceParams.major + deviceParams.minor / 10
print('Cuda compute capability: ' .. cudaComputeCapability)

local options = RNNOption()
local params = options:parse(arg)
params.vocab_size = 10000
local data = require('data')

local state_train, state_valid, state_test
local model = {}
local paramx, paramdx

local function lstm(x, prev_c, prev_h)
	-- Calculate all four gates in one go
	local i2h = nn.Linear(params.n_hidden, 4 * params.n_hidden)(x)
	local h2h = nn.Linear(params.n_hidden, 4 * params.n_hidden)(prev_h)
	local gates = nn.CAddTable()({i2h, h2h})
	
	-- Reshape to (batch_size, n_gates, hid_size)
	-- Then slize the n_gates dimension, i.e dimension 2
	local reshaped_gates =	nn.Reshape(4,params.n_hidden)(gates)
	local sliced_gates = nn.SplitTable(2)(reshaped_gates)
	
	-- Use select gate to fetch each gate and apply nonlinearity
	local in_gate = nn.Sigmoid()(nn.SelectTable(1)(sliced_gates))
	local in_transform = nn.Tanh()(nn.SelectTable(2)(sliced_gates))
	local forget_gate = nn.Sigmoid()(nn.SelectTable(3)(sliced_gates))
	local out_gate = nn.Sigmoid()(nn.SelectTable(4)(sliced_gates))

	local next_c = nn.CAddTable()({
		nn.CMulTable()({forget_gate, prev_c}),
		nn.CMulTable()({in_gate, in_transform})
	})
	local next_h = nn.CMulTable()({out_gate, nn.Tanh()(next_c)})

	return next_c, next_h
end

local function create_network()
	local x = nn.Identity()()
	local y = nn.Identity()()
	local prev_s = nn.Identity()()
	local i = {[0] = nn.LookupTable(params.vocab_size, params.n_hidden)(x)}
	local next_s = {}
	local split	= {prev_s:split(2 * params.layers)}

	for layer_idx = 1, params.layers do
		local prev_c = split[2 * layer_idx - 1]
		local prev_h = split[2 * layer_idx]
		local dropped = nn.Dropout(params.dropout)(i[layer_idx - 1])
		local next_c, next_h = lstm(dropped, prev_c, prev_h)
		table.insert(next_s, next_c)
		table.insert(next_s, next_h)
		i[layer_idx] = next_h
	end

	local h2y = nn.Linear(params.n_hidden, params.vocab_size)
	local dropped = nn.Dropout(params.dropout)(i[params.layers])
	local pred = nn.LogSoftMax()(h2y(dropped))
	local err = nn.ClassNLLCriterion()({pred, y})
	local module = nn.gModule({x, y, prev_s}, {err, nn.Identity()(next_s)})
	module:getParameters():uniform(-params.initial_weight, params.initial_weight)
	return module:cuda()
end

local function setup()
	print("Creating a RNN LSTM network.")
	local core_network = create_network()
	paramx, paramdx = core_network:getParameters()
	model.s = {}
	model.ds = {}
	model.start_s = {}
	for j = 0, params.seq_length do
		model.s[j] = {}
		for d = 1, 2 * params.layers do
			model.s[j][d] = torch.zeros(params.batch_size, params.n_hidden):cuda()
		end
	end
	for d = 1, 2 * params.layers do
		model.start_s[d] = torch.zeros(params.batch_size, params.n_hidden):cuda()
		model.ds[d] = torch.zeros(params.batch_size, params.n_hidden):cuda()
	end
	model.core_network = core_network
	model.rnns = g_cloneManyTimes(core_network, params.seq_length)
	model.norm_dw = 0
	model.err = torch.zeros(params.seq_length):cuda()
end

local function reset_state(state)
	state.pos = 1
	if model ~= nil and model.start_s ~= nil then
		for d = 1, 2 * params.layers do
			model.start_s[d]:zero()
		end
	end
end

local function reset_ds()
	for d = 1, #model.ds do
		model.ds[d]:zero()
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
	paramdx:zero()
	reset_ds()
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
	model.norm_dw = paramdx:norm()
	if model.norm_dw > params.max_grad_norm then
		local shrink_factor = params.max_grad_norm / model.norm_dw
		paramdx:mul(shrink_factor)
	end
	paramx:add(paramdx:mul(-params.learning_rate))
end

local function run_valid()
	reset_state(state_valid)
	g_disable_dropout(model.rnns)
	local len = (state_valid.data:size(1) - 1) / (params.seq_length)
	local perp = 0
	for i = 1, len do
		perp = perp + fp(state_valid)
	end
	print("\n")
	print("Validation set perplexity : " .. g_f3(torch.exp(perp / len)))
	g_enable_dropout(model.rnns)
end

local function run_test()
	reset_state(state_test)
	g_disable_dropout(model.rnns)
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
	g_enable_dropout(model.rnns)
end

function main()
	g_init_gpu(arg)

    if params.save_dir ~= nil then
       if paths.dirp(params.save_dir) == false then
           os.execute('mkdir -p ' .. params.save_dir)
       end
       print('*** models will be saved after each epoch ***')
    end

    datasets = load_datasets(params)
	state_train = {data=datasets.train:cuda()}
	state_valid = {data=datasets.valid:cuda()}
	state_test = {data=datasets.test:cuda()}
	print("Network parameters:")
	print(params)
	local states = {state_train, state_valid, state_test}
	for _, state in pairs(states) do
		reset_state(state)
	end
	setup()
	local step = 0
	local epoch = 0
	local total_cases = 0
	local beginning_time = torch.tic()
	local start_time = torch.tic()
	print("Starting training.")
	local words_per_step = params.seq_length * params.batch_size
	local epoch_size = torch.floor(state_train.data:size(1) / params.seq_length)
	local perps

	while epoch < params.n_epochs do
		local perp = forward_pass(state_train)
		if perps == nil then
			perps = torch.zeros(epoch_size):add(perp)
		end
		perps[step % epoch_size + 1] = perp
		step = step + 1
		backward_pass(state_train)
		total_cases = total_cases + params.seq_length * params.batch_size
		epoch = step / epoch_size
		xlua.progress(step % epoch_size, epoch_size)

		if step % 33 == 0 then
			cutorch.synchronize()
			collectgarbage()
		end

        -- End-of-epoch bookkeeping
		if step % epoch_size == 0 then
			local wps = torch.floor(total_cases / torch.toc(start_time))
			local since_beginning = g_d(torch.toc(beginning_time) / 60)
			print('epoch = ' .. torch.floor(epoch) ..
						', training perplexity = ' .. g_f3(torch.exp(perps:mean())) ..
						', word per second = ' .. wps ..
						', dw:norm() = ' .. g_f3(model.norm_dw) ..
						', learning_rate = ' ..  g_f3(params.learning_rate) ..
						', time elapsed = ' .. since_beginning .. ' mins.')
			run_valid()
			if epoch > params.n_epochs_before_decay then
				params.learning_rate = params.learning_rate / params.learning_rate_decay
			end
            if params.save_dir ~= nil then
                print 'Saving model to disk'
                local save_state = {}
                save_state.learning_rate = learning_rate
                save_state.learning_rate_decay = params.learning_rate_decay
                local filename = 'model_' .. torch.floor(epoch)
                torch.save(paths.concat(params.save_dir, filename), save_state)
            end
		end
	end
	run_test()
	print("Training is over.")
end
