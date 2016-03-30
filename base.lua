--
--	Copyright (c) 2014, Facebook, Inc.
--	All rights reserved.
--
--	This source code is licensed under the Apache 2 license found in the
--	LICENSE file in the root directory of this source tree. 
--

function g_cloneManyTimes(net, T)
	local clones = {}
	local params, gradParams = net:parameters()
	local mem = torch.MemoryFile("w"):binary()
	mem:writeObject(net)
	for t = 1, T do
	-- We need to use a new reader for each clone.
	-- We don't want to use the pointers to already read objects.
	local reader = torch.MemoryFile(mem:storage(), "r"):binary()
	local clone = reader:readObject()
	reader:close()
	local cloneParams, cloneGradParams = clone:parameters()
	for i = 1, #params do
		cloneParams[i]:set(params[i])
		cloneGradParams[i]:set(gradParams[i])
	end
	clones[t] = clone
	collectgarbage()
	end
	mem:close()
	return clones
end

function g_init_gpu(args)
	gpuidx = 1
	cutorch.setDevice(gpuidx)
	g_make_deterministic(1)
end

function g_make_deterministic(seed)
	torch.manualSeed(seed)
	cutorch.manualSeed(seed)
	torch.zeros(1, 1):cuda():uniform()
end

function g_replace_table(to, from)
	assert(#to == #from)
	for i = 1, #to do
	to[i]:copy(from[i])
	end
end

function g_f3(f)
	return string.format("%.3f", f)
end
