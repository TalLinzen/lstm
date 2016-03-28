--
--  Copyright (c) 2015, Facebook, Inc.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--
--  Author: Sumit Chopra <spchopra@fb.com>
--          Michael Mathieu <myrhev@fb.com>
--          Marc'Aurelio Ranzato <ranzato@fb.com>
--          Tomas Mikolov <tmikolov@fb.com>
--          Armand Joulin <ajoulin@fb.com>

-- This file contains a class RNNOption.
-- It parses the default options for RNNs and processes them.
-- Custom options can be added using option, optionChoice and
--   optionDisableIfNegative function.
-- The options are then parsed using the parse function.

require('os')
require('string')

local RNNOption = torch.class('RNNOption')

-- Init. Adds standard options.
function RNNOption:__init()
    self.cmd = torch.CmdLine()
    self.cmd.argseparator = '_'
    self.cmd:text()
    self.cmd:text('LSTM language model')
    self.cmd:text()
    self.cmd:text('Options:')

    self.options = {}

    self:option('-dset',
                'dataset', 'ptb',
                'Dataset')
    self:option('-nhid',
                'n_hidden', 200,
                'Number of hidden units')
    self:option('-init',
                'initial_weight', 0.1,
                'Weights are initialized to Uniform [-x, x]')
    self:option('-dropout',
                'dropout', 0.5,
                'Dropout probability')
    self:option('-sqlen',
                'seq_length', 20,
                'Number of steps to unfold back in time')
    self:option('-layers',
                'layers', 2,
                'Number of layers')
    self:option('-batchsz',
                'batch_size', 32,
                'Size of mini-batch')
    self:option('-lr',
                'learning_rate', 1,
                'Initial learning rate')
    self:option('-epochsbeforedecay',
                'n_epochs_before_decay', 4,
                'Start learning rate decay after this number of epochs')
    self:option('-decay',
                'learning_rate_decay', 1.2,
                'Learning rate decay')
    self:optionDisableIfNegative('-maxgradnorm',
                                 'max_grad_norm', 5,
                                 'Norm of gradient clipping (-1 to disable)')
    self:option('-nepochs',
                'n_epochs', 13,
                'Number of training epochs')
    self:option('-save',
                'save', true,
                'Whether to save the trained model or not')
end

-- Adds an option:
--  cmd_option: the command line option (eg. -eta)
--  param_name: the name of the option in lua (the parse function returns a
--    table with all options. This is the index of the option in this table).
--    It be specialized to a subtable using a dot (eg. trainer.learning_rate)
--  default: the default value
--  process: a function to be applied to the parameter
function RNNOption:option(cmd_option, param_name, default, help,
process_function)
    process_function = process_function or function(x) return x end
    self.cmd:option(cmd_option, default, help)
    local cmd_option_idx = cmd_option
    while cmd_option_idx:sub(1,1) == '-' do
        cmd_option_idx = cmd_option_idx:sub(2,-1)
    end
    self.options[param_name] = {cmd_option_idx, process_function}
end

-- Adds an option expecting a string. If the option is not in the list
-- <choices>, it raises an error.
function RNNOption:optionChoice(cmd_option, param_name, default, help, choices)
    local function f(x)
        for i = 1, #choices do
            if choices[i] == x then
                return x
            end
        end
        error('Option ' .. cmd_option .. ' cannot take value ' .. x
                  .. ' . Possible values are '
                  .. self:build_choices_string(choices))
    end
    self:option(cmd_option, param_name, default, help, f)
end

-- Adds an option expecting a number. It is replaced by nil if it is <= 0.
function RNNOption:optionDisableIfNegative(cmd_option, param_name, default,
                                           help)
    local function f(x)
        if x <= 0 then
            return nil
        else
            return x
        end
    end
    self:option(cmd_option, param_name, default, help, f)
end

-- Changes the default value to an option.
function RNNOption:change_default(cmd_option, new_default)
    if self.cmd.options[cmd_option] == nil then
        error('RNNOption: trying to change default, but option '
                  .. cmd_option .. ' does not exist')
    end
    self.cmd.options[cmd_option].default = new_default
end

function RNNOption:build_choices_string(choices)
    local out = '('
    for i = 1, #choices do
        if i ~= 1 then out = out .. '|' end
        out = out .. choices[i]
    end
    return out .. ')'
end

-- Parses the command line. It returns a table containing :
-- tables for the specialized options (eg. model, trainer, ...)
-- and the global parameters (eg. cuda_device)
function RNNOption:parse()
    local opt = self.cmd:parse(arg)
    local params = {}
    for k, v in pairs(self.options) do
        local cmd_option = v[1]
        local process_function = v[2]
        if k:find('.', 1, true) then
            local k1 = k:sub(1, k:find('.', 1, true)-1)
            local k2 = k:sub(k:find('.', 1, true)+1, -1)
            if params[k1] == nil then
                params[k1] = {}
            end
            params[k1][k2] = process_function(opt[cmd_option ])
        else
            params[k] = process_function(opt[cmd_option ])
        end
    end
    --
    -- save dir
    local function to_string(x)
        if x == nil then
            return 'nil'
        elseif type(x) == 'boolean' then
            if x then
                return 'true'
            else
                return 'false'
            end
        else
            return x
        end
    end

    local mdir = 
        'nepch=' .. params.n_epochs
        .. '_bsz=' .. params.batch_size
        .. '_init=' .. params.initial_weight
        .. '_layers=' .. params.layers
        .. '_dropout=' .. params.dropout
        .. '_nhid=' .. params.n_hidden
        .. '_sqlen=' .. params.seq_length
        .. '_lr=' .. params.learning_rate
        .. '_grdnrm=' .. to_string(params.max_grad_norm)

    local basedir = './output/' .. params.dataset
    if params.save == true then
        params.save_dir = paths.concat(basedir, mdir)
    else
        params.save_dir = nil
    end

    return params
end

-- prints the help
function RNNOption:text()
    self.cmd:text()
end

-- prints the value of the parameters <params>
function RNNOption:print_params(params)
    for k, v in pairs(params) do
        if type(v) == 'boolean' then
            if v then
                print('' .. k .. ': true')
            else
                print('' .. k .. ': false')
            end
        elseif type(v) ~= 'table' then
            print('' .. k .. ': ' .. v)
        end
    end
    for k, v in pairs(params) do
        if type(v) == 'table' then
            print('' .. k .. ':')
            for k2, v2 in pairs(v) do
                if type(v2) == 'boolean' then
                    if v2 then
                        print('  ' .. k2 .. ': true')
                    else
                        print('  ' .. k2 .. ': false')
                    end
                else
                    if type(v2) == 'table' then
                       print('  ' .. k2 .. ': table')
                    else
                       print('  ' .. k2 .. ': ' .. v2)
                    end
                end
            end
        end
    end
end
