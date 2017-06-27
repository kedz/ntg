local ntg = ntg or {}
ntg.network = {}

require('ntg.network.Network')
require('ntg.network.EmbeddingLookup')
require('ntg.network.LSTM')
require('ntg.network.StackedLSTM')
require('ntg.network.StackedBiLSTM')
require('ntg.network.MLPPredictor')
require('ntg.network.NoAttention')
require('ntg.network.ConcatAttention')
require('ntg.network.SoftMax')



