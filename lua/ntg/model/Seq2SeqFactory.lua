local Seq2SeqFactory = torch.class('ntg.model.Seq2SeqFactory')

function Seq2SeqFactory:createModelFromArgs(args, srcVocab, tgtVocab)

   local encoder = self:createEncoderFromArgs(args, srcVocab) 
   local decoder = self:createDecoderFromArgs(args, tgtVocab) 
   local bridge = self:createBridgeFromArgs(args)

   local model = ntg.model.Seq2Seq(encoder, bridge, decoder)
   return model
end

function Seq2SeqFactory:createEncoderFromArgs(args, srcVocab)
   
   local tokenSize 
   if args.enc_token_size > 0 then
      tokenSize = args.enc_token_size
   else
      assert(args.token_dim > 0)
      tokenSize = args.token_size
   end

   local rnnOutputSize
   if args.enc_rnn_out_size > 0 then
      rnnOutputSize = args.enc_rnn_out_size
   else
      assert(args.rnn_out_size > 0)
      rnnOutputSize = args.rnn_out_size
   end
   

   assert(srcVocab:size() > 0)
   local vocabSize = srcVocab:size()

   local cellTypes = {LSTM=true, GRU=true}
   local rnnCell
   if args.enc_rnn_cell ~= '' then
      assert(cellTypes[args.enc_rnn_cell])
      rnnCell = args.enc_rnn_cell
   else
      assert(cellTypes[args.rnn_cell])
      rnnCell = args.rnn_cell
   end

   local rnnLayers
   if args.enc_rnn_layers > 0 then
      rnnLayers = args.enc_rnn_layers
   else
      assert(args.rnn_layers > 0)
      rnnLayers = args.rnn_layers
   end

   local isBi = false
   if args.enc_bi_rnn == 'true' or args.enc_bi_rnn == 'TRUE' 
         or args.enc_bi_rnn == '1' then
      isBi = true
   end

   local contextOp = 'identity'
   if isBi then
      local validOps = {concat=1, sum=1, mean=1, forward=1, backward=1}
      assert(validOps[args.context_op])
      contextOp = args.context_op
   end

   local stateOp = 'identity'
   if isBi then
      local validOps = {concat=1, sum=1, mean=1, forward=1, backward=1}
      assert(validOps[args.state_op])
      stateOp = args.state_op
   end

   local embeddings = ntg.network.EmbeddingLookup(vocabSize, tokenSize)

   local rnn 

   if rnnCell == 'LSTM' then
      if isBi then
         if rnnLayers > 1 then
            -- Create Stacked Bi-LSTM
            rnn = ntg.network.StackedBiLSTM(rnnLayers,tokenSize,rnnOutputSize)
         else
            -- Create Bi-LSTM
            rnn = ntg.network.BiLSTM(rnnLayers, tokenSize, rnnOutputSize)
         end
      else
         if rnnLayers > 1 then
            -- Create Stacked LSTM
            rnn = ntg.network.StackedLSTM(rnnLayers, tokenSize, rnnOutputSize)
         else
            -- Create LSTM
            rnn = ntg.network.LSTM(tokenSize, rnnOutputSize)
         end
      end

   else
      print("IMPLEMENT GRU YOU DUMMY")
      os.exit()
   end

  
   local encoder = ntg.module.RNNEncoder(embeddings, rnn)

   return encoder

end

function Seq2SeqFactory:createDecoderFromArgs(args, tgtVocab)
   
   local tokenSize 
   if args.dec_token_size > 0 then
      tokenSize = args.dec_token_size
   else
      assert(args.token_size > 0)
      tokenSize = args.token_size
   end

   local rnnOutputSize
   if args.dec_rnn_out_size > 0 then
      rnnOutputSize = args.dec_rnn_out_size
   else
      assert(args.rnn_out_size > 0)
      rnnOutputSize = args.rnn_out_size
   end
   

   assert(tgtVocab:size() > 0)
   local vocabSize = tgtVocab:size()

   local cellTypes = {LSTM=true, GRU=true}
   local rnnCell
   if args.dec_rnn_cell ~= '' then
      assert(cellTypes[args.dec_rnn_cell])
      rnnCell = args.dec_rnn_cell
   else
      assert(cellTypes[args.rnn_cell])
      rnnCell = args.rnn_cell
   end

   local rnnLayers
   if args.dec_rnn_layers > 0 then
      rnnLayers = args.dec_rnn_layers
   else
      assert(args.rnn_layers > 0)
      rnnLayers = args.rnn_layers
   end

   local contextOp = 'identity'
   if isBi then
      local validOps = {concat=1, sum=1, mean=1, forward=1, backward=1}
      assert(validOps[args.context_op])
      contextOp = args.context_op
   end

   local stateOp = 'identity'
   if isBi then
      local validOps = {concat=1, sum=1, mean=1, forward=1, backward=1}
      assert(validOps[args.state_op])
      stateOp = args.state_op
   end

   local embeddings = ntg.network.EmbeddingLookup(vocabSize, tokenSize)

   local rnn 

   if rnnCell == 'LSTM' then
      if rnnLayers > 1 then
         -- Create Stacked LSTM
         rnn = ntg.network.StackedLSTM(rnnLayers, tokenSize, rnnOutputSize)
      else
         -- Create LSTM
         rnn = ntg.network.LSTM(tokenSize, rnnOutputSize)
      end

   else
      print("IMPLEMENT GRU YOU DUMMY")
      os.exit()
   end

   local decoder = ntg.module.RNNDecoder(embeddings, rnn)

   return decoder

end




function Seq2SeqFactory:createBridgeFromArgs(args)
   
   local encRNNOutputSize
   if args.enc_rnn_out_size > 0 then
      encRNNOutputSize = args.enc_rnn_out_size
   else
      assert(args.rnn_out_size > 0)
      encRNNOutputSize = args.rnn_out_size
   end

   local decRNNOutputSize
   if args.dec_rnn_out_size > 0 then
      decRNNOutputSize = args.dec_rnn_out_size
   else
      assert(args.rnn_out_size > 0)
      decRNNOutputSize = args.rnn_out_size
   end
 
   local cellTypes = {LSTM=true, GRU=true}
   local encRNNCell
   if args.enc_rnn_cell ~= '' then
      assert(cellTypes[args.enc_rnn_cell])
      encRNNCell = args.enc_rnn_cell
   else
      assert(cellTypes[args.rnn_cell])
      encRNNCell = args.rnn_cell
   end
   local decRNNCell
   if args.dec_rnn_cell ~= '' then
      assert(cellTypes[args.dec_rnn_cell])
      decRNNCell = args.dec_rnn_cell
   else
      assert(cellTypes[args.rnn_cell])
      decRNNCell = args.rnn_cell
   end

   local encRNNLayers
   if args.enc_rnn_layers > 0 then
      encRNNLayers = args.enc_rnn_layers
   else
      assert(args.rnn_layers > 0)
      encRNNLayers = args.rnn_layers
   end
   local decRNNLayers
   if args.dec_rnn_layers > 0 then
      decRNNLayers = args.dec_rnn_layers
   else
      assert(args.rnn_layers > 0)
      decRNNLayers = args.rnn_layers
   end

   local isBi = false
   if args.enc_bi_rnn == 'true' or args.enc_bi_rnn == 'TRUE' 
         or args.enc_bi_rnn == '1' then
      isBi = true
   end

   local stateOp = 'identity'
   if isBi then
      local validOps = {concat=1, sum=1, mean=1, forward=1, backward=1}
      assert(validOps[args.state_op])
      stateOp = args.state_op
   end
   
   local projOp = 'identity'
   local validOps = {identity=1, linear=1}
   assert(validOps[args.bridge_proj])
   projOp = args.bridge_proj


   print(encRNNOutputSize)
   print(decRNNOutputSize)
   print(encRNNCell)
   print(decRNNCell)
   print(encRNNLayers)
   print(decRNNLayers)


   local projectionInputSize = encRNNOutputSize
   if stateOp == 'concat' then 
      projectionInputSize = projectionInputSize * 2
   end
   local projectionOutputSize = decRNNOutputSize

   local bridgeCell = encRNNCell == 'LSTM' and decRNNCell == 'LSTM'

   local bridgeLayers = math.min(encRNNLayers, decRNNLayers)

   print("Bridge cell?", bridgeCell)
   print(bridgeLayers)

   if projOp == 'identity' then
      assert(projectionInputSize == projectionOutputSize,
         "Encoder state has different size than decoder state, " ..
         "projection required.")
   end
         

   local bridge = ntg.module.Bridge(stateOp, projOp, 
      projectionInputSize, projectionOutputSize, bridgeLayers, bridgeCell)

   print(bridge.network)

   return bridge

end

