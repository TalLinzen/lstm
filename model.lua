require('cunn')
require('nngraph')

local Model = torch.class('Model')


function Model:__init(seq_length, n_hidden, layers, batch_size, vocab_size,
        initial_weight, dropout)

    self.seq_length = seq_length
    self.n_hidden = n_hidden
    self.layers = layers
    self.batch_size = batch_size
    self.vocab_size = vocab_size
    self.initial_weight = initial_weight
    self.dropout = dropout

	self.core_network = self:create_network()
	self.paramx, self.paramdx = self.core_network:getParameters()
	self.s = {}
	self.ds = {}
	self.start_s = {}
	for j = 0, self.seq_length do
		self.s[j] = {}
		for d = 1, 2 * self.layers do
			self.s[j][d] = torch.zeros(self.batch_size, self.n_hidden):cuda()
		end
	end
	for d = 1, 2 * self.layers do
		self.start_s[d] = torch.zeros(self.batch_size, self.n_hidden):cuda()
		self.ds[d] = torch.zeros(self.batch_size, self.n_hidden):cuda()
	end

	self.rnns = g_cloneManyTimes(self.core_network, self.seq_length)
	self.norm_dw = 0
	self.err = torch.zeros(self.seq_length):cuda()
end


function Model:lstm(x, prev_c, prev_h)
	-- Calculate all four gates in one go
	local i2h = nn.Linear(self.n_hidden, 4 * self.n_hidden)(x)
	local h2h = nn.Linear(self.n_hidden, 4 * self.n_hidden)(prev_h)
	local gates = nn.CAddTable()({i2h, h2h})
	
	-- Reshape to (batch_size, n_gates, hid_size)
	-- Then slize the n_gates dimension, i.e dimension 2
	local reshaped_gates =	nn.Reshape(4, self.n_hidden)(gates)
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


function Model:create_network()
	local x = nn.Identity()()
	local y = nn.Identity()()
	local prev_s = nn.Identity()()
	local i = {[0] = nn.LookupTable(self.vocab_size, self.n_hidden)(x)}
	local next_s = {}
	local split	= {prev_s:split(2 * self.layers)}

	for layer_idx = 1, self.layers do
		local prev_c = split[2 * layer_idx - 1]
		local prev_h = split[2 * layer_idx]
		local dropped = nn.Dropout(self.dropout)(i[layer_idx - 1])
		local next_c, next_h = self:lstm(dropped, prev_c, prev_h)
		table.insert(next_s, next_c)
		table.insert(next_s, next_h)
		i[layer_idx] = next_h
	end

	local h2y = nn.Linear(self.n_hidden, self.vocab_size)
	local dropped = nn.Dropout(self.dropout)(i[self.layers])
	local pred = nn.LogSoftMax()(h2y(dropped))
	local err = nn.ClassNLLCriterion()({pred, y})
	local module = nn.gModule({x, y, prev_s}, {err, nn.Identity()(next_s)})
	module:getParameters():uniform(-self.initial_weight, self.initial_weight)
	return module:cuda()
end


function Model:reset_ds()
	for d = 1, #self.ds do
		self.ds[d]:zero()
	end
end


function Model:enable_dropout()
    _enable_node_dropout(self.rnns)
end


function Model:disable_dropout()
    _disable_node_dropout(self.rnns)
end


function _disable_node_dropout(node)
	if type(node) == "table" and node.__typename == nil then
        for i = 1, #node do
            node[i]:apply(_disable_node_dropout)
        end
        return
	end

	if string.match(node.__typename, "Dropout") then
        node.train = false
	end
end


function _enable_node_dropout(node)
	if type(node) == "table" and node.__typename == nil then
        for i = 1, #node do
            node[i]:apply(_enable_node_dropout)
        end
        return
	end

	if string.match(node.__typename, "Dropout") then
        node.train = true
	end
end

