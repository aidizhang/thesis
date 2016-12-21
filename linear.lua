require 'hdf5'
require 'nn'
require 'rnn'
require 'io'
require 'optim'

cmd = torch.CmdLine()

-- command args
cmd:option('--datafile', 'data/data.hdf5', 'data file')
cmd:option('--hidden_size', 30)
cmd:option('--embed_size', 50)
cmd:option('--epochs', 20)
cmd:option('--eta', 0.001)
cmd:option('--pretrain', true)
cmd:option('--use_saved', false)
cmd:option('--use_batch', true)
cmd:option('--minibatch_size', 16)
cmd:option('--batch_size', 10000)
cmd:option('--init_from', false)
cmd:option('--checkpoint_every', 10000)
cmd:option('--checkpoint_name', 'cv/checkpoint')
cmd:option('--run_gpu', false)

MAX_CONTEXT_LEN = 80
MAX_UTTERANCE_LEN = 40

function build_siamese_model(idx_to_embedding)
	if use_batch then
		local model = nn.Sequential()
		siamese = nn.ParallelTable()
		twin = nn.Sequential() -- input is matrix of dims (seq_len x minibatch_size)
		emb = nn.LookupTable(vocab_size, opt.embed_size) -- returns tensor of dims (seq_len x minibatch_size x embed_size)
		for i = 1,vocab_size do
			emb.weight[i] = idx_to_embedding[i]
			-- randomly initialize zero embeddings
			if emb.weight[i]:norm() < 1e-6 then
				emb.weight[i] = torch.randn(opt.embed_size)
			end
			emb.weight[i]:div(emb.weight[i]:norm()):div(embed_size)
		end

		-- create siamese
		-- twin:add(emb)
		twin:add(nn.SplitTable(1)) -- splits into seq_len table with entries of dims (minibatch_size x embed_size)
		twin:add(nn.CAddTable())

		-- add twins to siamese
		siamese:add(twin)
		siamese:add(twin:clone('weight','bias','gradWeight','gradBias'))

		-- create projection
		projection = nn.ParallelTable()
		projection:add(nn.Linear(opt.embed_size, opt.embed_size):noBias())
		projection:add(nn.Identity())

		-- build full model
		model:add(siamese) -- returns 2 tensors both of dims (minibatch_size x hidden_size)
		model:add(projection)
		model:add(nn.CMulTable()):add(nn.Sum(1, 1)) -- take pairwise dot products
		model:add(nn.Sigmoid()) -- apply sigmoid element-wise

		return model, emb
	else
		local model = nn.Sequential()
		-- create siamese
		siamese = nn.ParallelTable()
		twin = nn.Sequential() -- input is seq_len 1d vector (each row is word index)
		emb = nn.LookupTable(vocab_size, opt.embed_size) -- returns seq_len x embed_size tensor
		for i = 1,vocab_size do
			emb.weight[i] = idx_to_embedding[i]
			if emb.weight[i]:abs():sum() < 1e-6 then
				emb.weight[i] = torch.randn(opt.embed_size)
			end
			emb.weight[i]:div(emb.weight[i]:norm())
		end

		-- twin:add(emb)
		twin:add(nn.SplitTable(1)) -- splits into seq_len table with entries of dims (minibatch_size x embed_size)
		twin:add(nn.CAddTable())
		
		-- add twins to siamese
		siamese:add(twin)
		siamese:add(twin:clone('weight','bias','gradWeight','gradBias'))

		-- create projection
		projection = nn.ParallelTable()
		projection:add(nn.Linear(opt.embed_size, opt.embed_size):noBias())
		projection:add(nn.Identity())

		-- build full model
		model:add(siamese)
		model:add(projection)
		model:add(nn.CMulTable()):add(nn.Sum(1, 1)) -- take pairwise dot products
		model:add(nn.Sigmoid())
		return model, emb
	end
end

function recall(scores, k) 
	sorted, indices = torch.sort(scores)
	for i = 10,10-k+1,-1 do
		if indices[i] == 1 then
			return true
		end
	end
	return false
end

function main()
	opt = cmd:parse(arg)

	print("building model and criterion")
	-- get pretrained GloVe embeddings
	local f = hdf5.open(opt.datafile, 'r')
	local idx_to_embedding = f:read("idx_to_embedding"):all()
	vocab_size = idx_to_embedding:size(1)
	embed_size = opt.embed_size
	use_batch = opt.use_batch
	minibatch_size = opt.minibatch_size
	batch_size = opt.batch_size
	epochs = opt.epochs
	run_gpu = opt.run_gpu
	eta = opt.eta
	checkpoint_name = opt.checkpoint_name

	-- build glove model and criterion
	local model, lookup = build_siamese_model(idx_to_embedding)
	local criterion = nn.BCECriterion(nil, false)

	-- import test data
	local test_contexts = f:read("test_contexts"):partial({1,1000},{1,MAX_CONTEXT_LEN}):long()
	local test_utterances = f:read("test_utterances"):partial({1,1000},{1,MAX_UTTERANCE_LEN}):long()
	local test_all_distractors = f:read("test_all_distractors"):partial({1,9},{1,1000},{1,MAX_UTTERANCE_LEN}):long()
	print("test_contexts", test_contexts:size(1))
	print("test_utterances", test_utterances:size(1))
	print("test_all_distractors", test_all_distractors:size(1))
	print("test_all_distractors", test_all_distractors:size(2))

	-- transform test data to 1 index
	test_contexts:add(1)
	test_utterances:add(1)
	test_all_distractors:add(1)

	for epoch_num = 1,epochs do
		for batch_num = 0,torch.floor(1000000/batch_size)-1 do
			-- import train data
			print(string.format("importing train data from batch %d, epoch %d", batch_num, epoch_num))
			local train_contexts = f:read("train_contexts"):partial({batch_num*batch_size+1, (batch_num+1)*batch_size},{1,MAX_CONTEXT_LEN}):long()
			local train_utterances = f:read("train_utterances"):partial({batch_num*batch_size+1, (batch_num+1)*batch_size},{1,MAX_UTTERANCE_LEN}):long()
			local train_targets = f:read("train_targets"):partial({batch_num*batch_size+1, (batch_num+1)*batch_size})
			print("train_contexts", train_contexts:size(1))
			print("train_utterances", train_utterances:size(1))
			print("train_targets", train_targets:size(1))

			-- transform word indices to 1 index
			train_contexts:add(1)
			train_utterances:add(1)

			-- shuffle train data every batch
			local shuffle = torch.randperm(train_contexts:size(1))

			print("start load data")
			-- prepare data for model, splitting into num_chunks (624) minibatches
			if use_batch then
				num_chunks = torch.floor((train_contexts:size(1) - minibatch_size)/minibatch_size)
				print("num minibatches per batch/num chunks", num_chunks) -- num_chunks x minibatch_size ~= batch_size 
				x = {}
				y = {}
				for i = 1,num_chunks do
					local curcontexts = torch.Tensor(minibatch_size, MAX_CONTEXT_LEN):long()
					local curutterances = torch.Tensor(minibatch_size, MAX_UTTERANCE_LEN):long()
					local curtargets = torch.Tensor(minibatch_size):double()
					for j = 1,minibatch_size do
						local ind = (i-1) * minibatch_size + j
						-- if i == 1 then
						-- 	print("train_contexts i: " .. ind .. " " .. shuffle[ind])
						-- 	print(train_contexts:select(1,shuffle[ind]))
						-- 	print("train_utterances i: " .. ind .. " " .. shuffle[ind])
						-- 	print(train_utterances:select(1,shuffle[ind]))
						-- end
						curcontexts[j] = train_contexts:select(1,shuffle[ind])
						curutterances[j] = train_utterances:select(1,shuffle[ind])
						curtargets[j] = train_targets:narrow(1,shuffle[ind],1):double()
					end
					x[i] = {}
					x[i][1] = lookup:forward(curcontexts:t()):clone()
					x[i][2] = lookup:forward(curutterances:t()):clone()
					y[i] = curtargets
				end
			else
				num_chunks = train_contexts:size(1)
				x = {}
				y = {}
				for i = 1,num_chunks do 
					x[i] = {}
					x[i][1] = lookup:forward(train_contexts:select(1,shuffle[i])):clone()
					x[i][2] = lookup:forward(train_utterancses:select(1,shuffle[i])):clone()
					y[i] = train_targets:narrow(1,shuffle[i],1):double()
				end
			end

			print("start training batch")
			-- train minibatch
			local params, grad_params = model:getParameters()
			-- print("params " .. grad_params:size(1))
			local losssum = 0
			local losscnt = 0
			for i = 1,num_chunks do 
				local preds = model:forward(x[i])
				local loss = criterion:forward(preds, y[i])

				model:zeroGradParameters()
				local gradpreds = criterion:backward(preds, y[i])
				local finalgrad = model:backward(x[i], gradpreds)

				-- grad max norm
				local norm = grad_params:norm()
				local cureta = eta
				if norm > 5 then
					cureta = eta / norm * 5.0
				end

				model:updateParameters(cureta)
				-- local w = lookup.weight
				-- for j = 1,minibatch_size do
				-- 	local ind = (i-1) * minibatch_size + j
				-- 	local curcontexts = train_contexts:select(1,shuffle[ind])
				-- 	local curutterances = train_utterances:select(1,shuffle[ind])
				-- 	for k = 1,curcontexts:size(1) do
				-- 		w[curcontexts[k]] = w[curcontexts[k]] - finalgrad[1][k][j] * cureta
				-- 	end
				-- 	for k = 1,curutterances:size(1) do
				-- 		w[curutterances[k]] = w[curutterances[k]] - finalgrad[2][k][j] * cureta
				-- 	end
				-- end

				-- print stats
				losssum = losssum + loss
				losscnt = losscnt + 1
				if losscnt == 50 then
					print(string.format("loss : %f", losssum / losscnt))
					print(preds[1])
					print(y[i][1])
					print(norm)
					losssum = 0
					losscnt = 0
				end
			end

			print("start testing")
			-- test
			local correct1 = 0
			local correct2 = 0
			local correct5 = 0
			for i = 1,test_contexts:size(1) do
				-- keep track of 10 scores
				scores = torch.Tensor(10)

				if use_batch then
					x = {}
					local curcontexts = torch.Tensor(10, MAX_CONTEXT_LEN)
					local curutterances = torch.Tensor(10, MAX_UTTERANCE_LEN)
					-- make batch input of batch_size 10, where each context is repeated text context
					for j = 1,10 do
						if j == 1 then
							curcontexts[j] = test_contexts:select(1,i)
							curutterances[j] = test_utterances:select(1,i)
						else
							curcontexts[j] = test_contexts:select(1,i)
							curutterances[j] = test_all_distractors:select(1,j-1):select(1,i)
						end
					end
					x[1] = lookup:forward(curcontexts:t()):clone()
					x[2] = lookup:forward(curutterances:t()):clone()
					scores = model:forward(x)
				else
					x = {}
					x[1] = lookup:forward(test_contexts:select(1,i)):clone()
					-- score all ground truth utterance/distractors using model
					for j = 1,10 do
						if j == 1 then
							x[2] = lookup:forward(test_utterances:select(1,i)):clone()
						else
							x[2] = lookup:forward(test_all_distractors:select(1,j-1):select(1,i)):clone()
						end
						scores[j] = model:forward(x)
					end
				end
				-- calculate various recall metrics
				if recall(scores, 1) then
					correct1 = correct1 + 1
				end
				if recall(scores, 2) then
					correct2 = correct2 + 1
				end
				if recall(scores, 5) then
					correct5 = correct5 + 1
				end
			end
			print("recall@1 " .. correct1/test_contexts:size(1))
			print("recall@2 " .. correct2/test_contexts:size(1))
			print("recall@5 " .. correct5/test_contexts:size(1))
		end
	end
end

main()