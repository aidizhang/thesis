require 'hdf5'
require 'nn'
require 'rnn'
require 'io'
require 'optim'

cmd = torch.CmdLine()

-- command args
cmd:option('--datafile', 'data/data.hdf5', 'data file')
cmd:option('--hidden_size', 64)
cmd:option('--batch_size', 1000)
cmd:option('--embed_size', 50)
cmd:option('--epochs', 20)
cmd:option('--eta', 0.01)

function build_glove_model(idx_to_embedding)
	model = nn.Sequential()
	-- create siamese
	siamese = nn.ParallelTable()
	twin = nn.Sequential()
	emb = nn.LookupTable(vocab_size, embed_size)
	for i = 1,vocab_size do
		emb.weight[i] = idx_to_embedding[i]
		if emb.weight[i]:norm() < 1e-6 then
			emb.weight[i] = torch.randn(opt.embed_size)
		end
		emb.weight[i]:div(emb.weight[i]:norm())
	end
	twin:add(emb)
	twin:add(nn.Mean(1))
	siamese:add(twin)
	siamese:add(twin:clone('weight','bias','gradWeight','gradBias'))
	
	-- add siamese and subsequent layers
	model:add(siamese)
	model:add(nn.PairwiseDistance())
	return model
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
	batch_size = opt.batch_size
	embed_size = opt.embed_size
	eta = opt.eta

	print("building model and criterion")
	-- get pretrained GloVe embeddings
	local f = hdf5.open(opt.datafile, 'r')
	local idx_to_embedding = f:read("idx_to_embedding"):all()
	vocab_size = idx_to_embedding:size(1)

	-- build glove model and criterion
	local model = build_glove_model(idx_to_embedding)
	local criterion = nn.ClassNLLCriterion()

	-- import test data
	local test_contexts = f:read("test_contexts"):all():long()
	local test_utterances = f:read("test_utterances"):all():long()
	local test_all_distractors = f:read("test_all_distractors"):all():long()
	print("test_contexts", test_contexts:size(1))
	print("test_utterances", test_utterances:size(1))
	print("test_all_distractors", test_all_distractors:size(1))
	print("test_all_distractors", test_all_distractors:size(2))

	-- transform test data to 1 index
	test_contexts:add(1)
	test_utterances:add(1)
	for i = 1,9 do
		test_all_distractors:select(1,i):add(1)
	end

	-- import train data
	for batch_num = 0,20 do
		print("importing data from batch number", batch_num)
		local train_contexts = f:read("train_contexts"):partial({batch_num*batch_size+1, (batch_num+1)*batch_size},{1,160}):long()
		local train_utterances = f:read("train_utterances"):partial({batch_num*batch_size+1, (batch_num+1)*batch_size},{1,80}):long()
		local train_targets = f:read("train_targets"):partial({batch_num*batch_size+1, (batch_num+1)*batch_size})
		print("train_contexts", train_contexts:size(1))
		print("train_utterances", train_utterances:size(1))
		print("train_targets", train_targets:size(1))

		-- transform word indices to 1 index
		train_contexts:add(1)
		train_utterances:add(1)

		print("start load data")
		-- prepare data for model
		local x = {}
		local y = {}
		for i = 1,train_contexts:size(1) do 
			x[i] = {}
			x[i][1] = train_contexts:select(1,i)
			x[i][2] = train_utterances:select(1,i)
			y[i] = train_targets:narrow(1,i,1):double()
		end

		print("start training mini-batch")
		-- train batch
		local params, grad_params = model:getParameters()
		print("params " .. grad_params:size(1))
		local losssum = 0
		local losscnt = 0
		for i = 1,batch_size do 
			-- forward pass
			local preds = model:forward(x[i])
			local loss = criterion:forward(preds, y[i])

			-- backward pass
			model:zeroGradParameters()
			local gradpreds = criterion:backward(preds, y[i])
			local finalgrad = model:backward(x[i], gradpreds)

			-- grad max normalization
			local norm = grad_params:norm()
			if norm > 5 then
				grad_params:div(norm):mul(5)
			end
			model:updateParameters(opt.eta)

			-- print stats
			if losscnt == 0 then
				print("preds")
				print(preds)
			end

			losssum = losssum + loss
			losscnt = losscnt + 1
			if losscnt == 50 then
				print(string.format("loss : %f", losssum / losscnt))
				losssum = 0
				losscnt = 0
			end
		end
		print(string.format("average batch loss : %f", batch_loss / train_contexts:size(1)))

		print("start testing")
		-- test
		local correct1 = 0
		local correct2 = 0
		local correct5 = 0
		for i = 1,test_contexts:size(1) do
			scores = torch.Tensor(10)
			local x = {}
			x[1] = test_contexts:select(1,i)
			-- score all ground truth utterance/distractors using model
			for j = 1,10 do
				if j == 1 then
					x[2] = test_utterances:select(1,i)
				else
					x[2] = test_all_distractors:select(1,j-1):select(1,i)
				end
				scores[j] = model:forward(x)
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
		print(scores)
		print("recall@1 " .. correct1/test_contexts:size(1))
		print("recall@2 " .. correct2/test_contexts:size(1))
		print("recall@5 " .. correct5/test_contexts:size(1))
	end
end

main()