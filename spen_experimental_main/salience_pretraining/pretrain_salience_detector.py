import argparse
import random 
import torch


import nt


from dataio.json_reader import JSONReader
from dataio.reader.label_reader import LabelReader2
from dataio.reader.dense_real_vector_reader import DenseRealVectorReader
from dataset import Dataset3



import dataio
from dataio.reader.json_salience_pretrain_reader import JsonSaliencePretrainReader
from dataio.recipes.summarization import read_salience_pretraining_data
import trainer

import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self, embedding_dim, dropout=.75):
        super(Model, self).__init__()
        self.dropout_ = dropout
        self.embedding_dim_ = embedding_dim
        self.linear = nn.Linear(embedding_dim, 1)
        #self.hidden_layer = nn.Linear(100, 1)

    def forward(self, inputs):
        inputs = F.dropout(inputs, p=self.dropout_, training=self.training)
        output = self.linear(inputs).squeeze()
        #hidden = F.relu(self.linear(inputs))
        #output = self.hidden(hidden).squeeze()
        return output



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument(
        "--embedding-size", type=int, required=False, default=300)

    parser.add_argument(
        "--gpu", default=-1, type=int, required=False)
    parser.add_argument(
        "--epochs", default=25, type=int, required=False)
    parser.add_argument(
        "--seed", default=83432534, type=int, required=False)

#    parser.add_argument(
#        "--optimizer", default="adagrad", type=str, required=False,
#        choices=["sgd", "adagrad", "adadelta", "adam"])
    parser.add_argument(
        "--lr", required=False, default=.001, type=float)
    parser.add_argument(
        "--batch-size", default=16, type=int, required=False)
    parser.add_argument(
        "--dropout", default=0, type=float, required=False)

    parser.add_argument(
        "--save-model", default=None, required=False, type=str)
    parser.add_argument(
        "--save-results", default=None, required=False, type=str)

    args = parser.parse_args()

    random.seed(args.seed)
    #torch.manual_seed(args.seed)

    model = Model(args.embedding_size, dropout=args.dropout)
    import optimizer
    print(optimizer)
    print(optimizer)
    opt = optimizer.Adam(model.parameters(), args.lr,
        weight_decay=0.0)


    feature_reader = DenseRealVectorReader.field_reader(
        "dict", "embedding", args.embedding_size)
    target_reader = LabelReader2.field_reader("dict", "label")
    data_reader = JSONReader([feature_reader, target_reader])
    data_reader.fit_parameters(args.data)
    (features,), (targets,) = data_reader.read(args.data)
   
    dataset = Dataset3((features, "inputs"), (targets.float(), "targets"),
                       batch_size=args.batch_size, shuffle=True, gpu=-1)
 
    print(target_reader.vocab_.count)
    print(target_reader.vocab_.index2token_)
    weight = [1 / target_reader.vocab_.count[label] 
              for label in target_reader.vocab_.index2token_]
    weight = torch.FloatTensor(weight) #* 10000
    print(weight)


    tr_idx, val_idx, te_idx = nt.trainer.stratified_generate_splits(
        [i for i in range(dataset.size)], dataset.targets.long())

    data_train = dataset.index_select(tr_idx)
    data_valid = dataset.index_select(val_idx)
    
    crit = nt.criterion.BinaryCrossEntropy(mode="logit", weight=weight)
    crit.add_reporter(nt.criterion.BinaryAccuracyReporter(mode="logit"))
    crit.add_reporter(nt.criterion.BinaryFMeasureReporter(mode="logit"))
    
    nt.trainer.optimize_criterion(
        crit, model, opt, data_train, validation_data=data_valid,
        max_epochs=args.epochs)

    


    exit()
    #crit = criterion.PrecRecallReporter(crit, label_reader.vocab.index2token_)
    results = trainer.minimize_criterion(
        crit, data_train, data_valid, args.epochs,
        save_best_model=args.save_model)





    reader = JsonSaliencePretrainReader()
    data_train = read_salience_pretraining_data(
        args.train, reader, batch_size=args.batch_size, gpu=-1)
    data_valid = data_train    


    if args.gpu > -1:
        with torch.cuda.device(args.gpu):
            torch.cuda.manual_seed(args.seed)
        model = model.cuda(args.gpu)

    if args.optimizer == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
          #  weight_decay=args.l2)
    elif args.optimizer == "adagrad":
        optimizer = torch.optim.Adagrad(model.parameters(), lr=args.lr)
        #    weight_decay=args.l2)
    elif args.optimizer == "adadelta":
        optimizer = torch.optim.Adadelta(model.parameters(), lr=args.lr)
         #   weight_decay=args.l2)
    elif args.optimizer == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
         #   weight_decay=args.l2)
    else:
        raise Exception("Unkown optimizer: {}".format(args.optimizer))

    if args.save_model is not None:
        d = os.path.dirname(args.save_model)
        if d != "" and not os.path.exists(d):
            os.makedirs(d)

    weight=None
    #weight = torch.FloatTensor(
    #    [1., 2.9964664310954063, 3.129151291512915]).cuda(1)

    crit = criterion.BinaryCrossEntropy(model, optimizer, weight=weight)
    #crit = criterion.PrecRecallReporter(crit, label_reader.vocab.index2token_)
    results = trainer.minimize_criterion(
        crit, data_train, data_valid, args.epochs,
        save_best_model=args.save_model)

    best_valid_loss = min(results["valid_nll"])
    best_epoch = results["valid_nll"].index(best_valid_loss) + 1
    print("\nBest epoch: {}".format(best_epoch))
    print("Best validation nll: {}".format(best_valid_loss))





if __name__ == "__main__":
    main()
