import argparse
import time

import dataio
from dataio.reader.mds_embedding_reader import MDSEmbeddingReader
from dataio.recipes.summarization import read_mds_embedding_data
import torch
from torch.autograd import Variable
from torch.nn import Parameter

from Utils import objectview
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


# from torch.utils.data import DataLoader
# from Utils import Dataset
dtype = torch.cuda.FloatTensor  # Uncomment this to run on GPU

class SalienceFunction(nn.Module):

    def __init__(self, input_size):
        '''
        input_size: sentence embedding dimension
        '''
        super(SalienceFunction, self).__init__()

        self.linear = nn.Linear(input_size, 1)

    # TODO: better flatten the tensor
    def forward(self, inputs, targets):
        # for now we have used RELU but it can be generalized
        # returns the Salience energy for all documents in the batch.
        _sz = inputs.size()
        return targets * F.relu(self.linear(inputs).view(_sz[0], _sz[1]))

class DocumentCentrality(nn.Module):

    def __init__(self, input_size, use_identity = False):
        '''
        input_size: sentence embedding dimension
        use_identity: If True, Identity matrix will be used instead of U
        '''
        super(DocumentCentrality, self).__init__()
        if use_identity:
            # TODO: verify that this works!
            self.W = Variable(torch.eye(input_size).type(dtype), requires_grad = False)
        else:
#             self.weights = Parameter(torch.zeros(2, 1),
#                                  requires_grad=True)
            self.W = Parameter(torch.rand(input_size, input_size))
            # self.W = Variable(torch.rand(input_size, input_size).type(dtype), requires_grad = True)
        # bias term
        # self.b = Variable(torch.rand(1, 1).type(dtype), requires_grad = True)
        self.b = Parameter(torch.rand(1, 1))

    def forward(self, inputs, targets):
        v1 = torch.matmul(inputs, self.W)
        v2 = -(inputs - (torch.sum(inputs, dim = 1).view(inputs.size(0), 1, inputs.size(2))))
        v = v1 * v2
        v = torch.sum(v, dim = 2)
        # At this point, v is (batch_size, n_sentences)
        v = v + self.b
        v = F.relu(v)
        v = targets * v
        # At this point, v is (batch_size, n_sentences)
        return v

class SummaryCoverage(nn.Module):
    def __init__(self, input_size, use_identity = False):
        '''
        input_size: sentence embedding dimension
        use_identity: If True, Identity matrix will be used instead of U
        '''
        super(SummaryCoverage, self).__init__()
        if use_identity:
            # TODO: verify that this works!
            self.W = Variable(torch.eye(input_size).type(dtype), requires_grad = False)
        else:
            self.W = Parameter(torch.rand(input_size, input_size))
            # self.W = Variable(torch.rand(input_size, input_size).type(dtype), requires_grad = True)
        # bias term
        # self.b = Variable(torch.rand(1, 1).type(dtype), requires_grad = True)
        self.b = Parameter(torch.rand(1, 1))

    def forward(self, inputs, targets):
        v1 = torch.matmul(inputs, self.W)
        v2 = (1 - targets).view(targets.size(0), targets.size(1), 1) * inputs
        v3 = torch.sum(v2, dim = 1)
        v3 = v3.view(v3.size(0), 1, v3.size(2))
        v2 = -(v2 - v3)
        v = v1 * v2
        v = torch.sum(v, dim = 2)
        # At this point, v is (batch_size, n_sentences)
        v = v + self.b
        v = F.relu(v)
        v = -(targets * v)
        # At this point, v is (batch_size, n_sentences)
        return v

class SummaryDiversity(nn.Module):
    def __init__(self, input_size, use_identity = False):
        '''
        input_size: sentence embedding dimension
        use_identity: If True, Identity matrix will be used instead of U
        '''
        super(SummaryDiversity, self).__init__()
        if use_identity:
            # TODO: verify that this works!
            self.W = Variable(torch.eye(input_size).type(dtype), requires_grad = False)
        else:
            self.W = Parameter(torch.rand(input_size, input_size))
            # self.W = Variable(torch.rand(input_size, input_size).type(dtype), requires_grad = True)
        # bias term
        self.b = Parameter(torch.rand(1, 1))
        # self.b = Variable(torch.rand(1, 1).type(dtype), requires_grad = True)

    def forward(self, inputs, targets):
        v1 = torch.matmul(inputs, self.W)
        v2 = targets.view(targets.size(0), targets.size(1), 1) * inputs
        v3 = torch.sum(v2, dim = 1)
        v3 = v3.view(v3.size(0), 1, v3.size(2))
        v2 = -(v2 - v3)
        v = v1 * v2
        v = torch.sum(v, dim = 2)
        # At this point, v is (batch_size, n_sentences)
        v = v + self.b
        v = F.relu(v)
        v = targets * v
        # At this point, v is (batch_size, n_sentences)
        return v

class EnergySummarizer(nn.Module):

    def __init__(self, kwargs):
        '''
        kwargs: object holding hyper-parameters
        '''
        super(EnergySummarizer, self).__init__()
        self.kwargs = kwargs
        self.params = []
        self.initialize_predicted_labels()
        self.salience_func = SalienceFunction(kwargs.input_size)
        for param in self.salience_func.parameters():
            if not isinstance(param, Variable):
                print (param)
            self.params.append(param)
        # self.document_centrality_func = DocumentCentrality(kwargs.input_size)
#         for param in self.document_centrality_func.parameters():
#             if not isinstance(param, Variable):
#                 print (param)
#             self.params.append(param)
        # TODO: add learning rate decay schedule
        # optimizer.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)
        # self.opt_loss = optim.SGD([self.salience_func.parameters(), self.document_centrality_func.parameters()], lr = kwargs.learning_rate)
#
#
        self.opt_loss = optim.SGD(self.params, lr = kwargs.learning_rate)

    def initialize_predicted_labels(self):
        # predicted labels: (batch_size, n_sentences)
        self.predicted_labels = Variable(0.5 * torch.ones(self.kwargs.batch_size, self.kwargs.n_sentences).type(dtype), requires_grad = True)
        self.opt_plabels = optim.SGD([self.predicted_labels], lr = self.kwargs.prediction_learning_rate)

    # TODO: make the loss different for train and test
    def fit_predicted_labels(self, inputs):
        iter = 0
        while iter < self.kwargs.iterations:
            self.opt_plabels.zero_grad()
            p_energy = self.compute_energy(inputs, self.predicted_labels)
            total_p_energy = torch.sum(p_energy)
            total_p_energy.backward()
            self.opt_plabels.step()
            # clip the labels between [0,1]
            self.predicted_labels = torch.clamp(self.predicted_labels, min = 0.0, max = 1.0)
            iter += 1

    def round_predicted_labels(self):
        # round off the labels so that they are either 0 or 1.
        self.predicted_labels = torch.round(self.predicted_labels)

    def compute_energy(self, inputs, targets):
        salience_energy = self.salience_func(inputs, targets)
        total_energy = salience_energy
        return total_energy

    def compute_loss(self, p_labels, g_labels, p_energy, g_energy):

        delta = torch.pow(p_labels - g_labels, 2)
        total_loss = delta + g_energy - p_energy
        _sz = total_loss.size()
        total_loss = torch.max(total_loss, Variable(torch.zeros((_sz[0], _sz[1])).type(dtype)))
        total_loss = torch.sum(total_loss)
        return total_loss

    def forward(self, inputs, targets):
        '''
        we are directly using sentence representations so this function just
        computes the energy given the sentence representations
        inputs: (batch_size, n_sentences, embedding_dimension)
        targets: (batch_size, n_sentences)
        This method does computation per batch per epoch
        '''
        # re-initialize the predicted labels
        self.initialize_predicted_labels()
        # fit the predicted labels
        self.fit_predicted_labels(inputs)
        # round off the predicted labels
        self.round_predicted_labels()
        # compute predicted energies
        p_energy = self.compute_energy(inputs, self.predicted_labels)
        # compute gold energies
        g_energy = self.compute_energy(inputs, targets)
        loss = self.compute_loss(self.predicted_labels, targets, p_energy, g_energy)
        return loss, torch.sum(p_energy), torch.sum(g_energy)

def get_variables(data, target, evaluation = False):
    data = autograd.Variable(data.cuda(), volatile = evaluation)
    target = autograd.Variable(target.cuda())
    return data, target

def fit(model, data_loader):

    train_loss = 0
    t_p_energy = 0
    t_g_energy = 0
    tp = 0
    tp_fp = 0
    tp_fn = 0

    for batch_num, batch in enumerate(data_loader.iter_batch()):
        # print("Batch {}".format(batch_num))
        # At this point
        # data: (batch_size, n_sentences, embedding_dimension)
        # target: (batch_size, n_sentences)
        target = batch.targets.data.type(torch.FloatTensor)
        data = batch.inputs.data

        # update n_sentence length for each batch
        model.kwargs.n_sentences = data.size()[1]

        model.opt_loss.zero_grad()

        data, target = get_variables(data, target)

        loss, p_energy, g_energy = model(data, target)

        loss.backward()

        model.opt_loss.step()

        # update tp, tp_fp, tp_fn
        # cuda variable to numpy array: (Variable(x).data).cpu().numpy()
        tp += torch.sum(model.predicted_labels.data * target.data)
        tp_fp += torch.sum(model.predicted_labels.data)
        tp_fn += torch.sum(target.data)

        train_loss += loss.data[0]
        t_p_energy += p_energy.data[0]
        t_g_energy += g_energy.data[0]

    precision = float(tp) / tp_fp
    recall = float(tp) / tp_fn
    f1 = (2 * recall * precision) / (recall + precision)

    return train_loss, t_p_energy, t_g_energy, precision, recall, f1

def read_data():
    pass

def main(args):
    '''
    X_train: (num_sentences,embedding_dim)
    y_train: (num_sentences,)
    n_sentences: number of sentences per document. Assuming it will be constant across all documents.
    '''

    # batch size is multiple of sentences_per_document
    batch_size = 2

    reader = MDSEmbeddingReader()
    data_loader = read_mds_embedding_data(
        args['train_file'], reader, batch_size = batch_size, gpu = -1)

    print (args)

    # TODO: define the hyper parameters here
    args['batch_size'] = batch_size
    args['n_sentences'] = -1  # We are going to change this for each batch so it does not matter whatever value we assign here.
    args['iterations'] = 30  # number of iterations for fitting predicted labels
    args['input_size'] = 300  # sentence embedding dimension

#     print(type(args))
#     print (args)
    kwargs = objectview(args)
#
#     print(kwargs)
#
#     print (kwargs.batch_size)
#     print (kwargs.n_sentences)
#     print (kwargs.iterations)
#     print (kwargs.input_size)

    summarizer = EnergySummarizer(kwargs)
    # TODO: add a flag to use or not use CUDA
    summarizer = summarizer.cuda()

    train_losses = []

    for epoch in range(1, args['n_epochs'] + 1):
        epoch_start_time = time.time()
        train_loss, p_energy, g_energy, precision, recall, f1 = fit(summarizer, data_loader)
        train_losses.append(train_loss)
        print('| Epoch {:3d} | training time: {:5.2f}s |'.format(epoch, (time.time() - epoch_start_time)))
        print('Gold energy: %s, Predicted energy: %s' % (g_energy, p_energy))
        print('Epoch %s: recall: %s, precision: %s, f1(%s) loss(%s)' % (epoch, recall, precision, f1, train_loss))

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description = '')

    parser.add_argument('-tr', "--train_file", type = str, required = True)
    parser.add_argument('-msd', '--model_save_dir', type = str, required = True)
    parser.add_argument('--n_epochs', type = int, default = 20)
    parser.add_argument('-lr', '--learning_rate', type = float, default = 0.005)
    parser.add_argument('-lr_plabels', '--prediction_learning_rate', type = float, default = 0.1)

    main(vars(parser.parse_args()))
