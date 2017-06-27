require('torch')
require('nn')
require('dpnn')
require('rnn')

ntg = {}

ntg.util = require('ntg.util.init')
ntg.modules = require('ntg.modules.init')
ntg.data = require('ntg.data.init')
require('ntg.Seq2SeqModel')
require('ntg.Trainer')

require('ntg.network.init')
require('ntg.module.init')
require('ntg.model.init')



--
--onmt.data = require('onmt.data.init')
--onmt.train = require('onmt.train.init')
--onmt.translate = require('onmt.translate.init')
--onmt.tagger = require('onmt.tagger.init')

--onmt.Constants = require('onmt.Constants')
--onmt.Factory = require('onmt.Factory')
--onmt.Model = require('onmt.Model')
--onmt.LanguageModel = require('onmt.LanguageModel')
--onmt.SeqTagger = require('onmt.SeqTagger')
--onmt.ModelSelector = require('onmt.ModelSelector')

return ntg
