import unittest
from nt.criterion import BinaryCrossEntropy
import torch
from torch.autograd import Variable
import math

class TestBinaryCrossEntropy(unittest.TestCase):

    def test_binary_cross_entropy_prob_weight_mask(self):
        tol = 1e-5

        prob = Variable(torch.rand(9).float())
        target = Variable(torch.FloatTensor([1, 0, -1,  1, -1, 0, -1, 1, 0]))
        weight = torch.FloatTensor([.8, .2])

        bce = BinaryCrossEntropy(mode="prob", weight=weight, mask_value=-1)

        total_nll = 0
        count = 0
        for i in range(0, 9, 3):
            batch_nll = 0

            for j in range(i, i + 3):
                if target.data[j] == 1:
                    batch_nll -= weight[1] * torch.log(prob[j:j+1]).data[0]
                    count += 1
                elif target.data[j] == 0:
                    batch_nll -= weight[0] \
                        * torch.log(1 - prob[j:j+1]).data[0]
                    count += 1
                    
            total_nll += batch_nll
            exp_batch_loss = batch_nll / 2
            batch_loss = bce.eval(prob[i:i+3], target[i:i+3]).data[0]
            self.assertTrue(abs(batch_loss - exp_batch_loss) < tol)

            exp_avg_loss = total_nll / count
            self.assertTrue(abs(exp_avg_loss - bce.avg_loss) < tol)

        bce.reset()
        self.assertTrue(math.isnan(bce.avg_loss))

    def test_binary_cross_entropy_logit_weight_mask(self):
        tol = 1e-5
        
        logit = Variable(torch.rand(9).float() - .5)
        target = Variable(torch.FloatTensor([1, 0, -1,  1, -1, 0, -1, 1, 0]))
        weight = torch.FloatTensor([.8, .2])

        bce = BinaryCrossEntropy(mode="logit", weight=weight, mask_value=-1)
        
        total_nll = 0
        count = 0
        for i in range(0, 9, 3):
            batch_nll = 0

            for j in range(i, i + 3):
                if target.data[j] == 1:
                    batch_nll -= weight[1] * torch.log(
                        1 / (1 + torch.exp(-logit[j:j+1]))).data[0]
                    count += 1
                elif target.data[j] == 0:
                    batch_nll -= weight[0] * torch.log(
                        1 - 1 / (1 + torch.exp(-logit[j:j+1]))).data[0]
                    count += 1
                    
            total_nll += batch_nll
            exp_batch_loss = batch_nll / 2
            batch_loss = bce.eval(logit[i:i+3], target[i:i+3]).data[0]
            self.assertTrue(abs(batch_loss - exp_batch_loss) < tol)

            exp_avg_loss = total_nll / count
            self.assertTrue(abs(exp_avg_loss - bce.avg_loss) < tol)

        bce.reset()
        self.assertTrue(math.isnan(bce.avg_loss))

    def test_binary_cross_entropy_prob_mask(self):
        tol = 1e-5

        bce = BinaryCrossEntropy(mode="prob", mask_value=-1)
        
        prob = Variable(torch.rand(9).float())
        target = Variable(torch.FloatTensor([1, 0, -1,  1, -1, 0, -1, 1, 0]))
        
        total_nll = 0
        count = 0
        for i in range(0, 9, 3):
            batch_nll = 0

            for j in range(i, i + 3):
                if target.data[j] == 1:
                    batch_nll -= torch.log(prob[j:j+1]).data[0]
                    count += 1
                elif target.data[j] == 0:
                    batch_nll -= torch.log(1 - prob[j:j+1]).data[0]
                    count += 1
                    
            total_nll += batch_nll
            exp_batch_loss = batch_nll / 2
            batch_loss = bce.eval(prob[i:i+3], target[i:i+3]).data[0]
            self.assertTrue(abs(batch_loss - exp_batch_loss) < tol)

            exp_avg_loss = total_nll / count
            self.assertTrue(abs(exp_avg_loss - bce.avg_loss) < tol)

        bce.reset()
        self.assertTrue(math.isnan(bce.avg_loss))

    def test_binary_cross_entropy_logit_mask(self):
        tol = 1e-5

        bce = BinaryCrossEntropy(mode="logit", mask_value=-1)
        
        logit = Variable(torch.rand(9).float() - .5)
        target = Variable(torch.FloatTensor([1, 0, -1,  1, -1, 0, -1, 1, 0]))
        
        total_nll = 0
        count = 0
        for i in range(0, 9, 3):
            batch_nll = 0

            for j in range(i, i + 3):
                if target.data[j] == 1:
                    batch_nll -= torch.log(
                        1 / (1 + torch.exp(-logit[j:j+1]))).data[0]
                    count += 1
                elif target.data[j] == 0:
                    batch_nll -= torch.log(
                        1 - 1 / (1 + torch.exp(-logit[j:j+1]))).data[0]
                    count += 1
                    
            total_nll += batch_nll
            exp_batch_loss = batch_nll / 2
            batch_loss = bce.eval(logit[i:i+3], target[i:i+3]).data[0]
            self.assertTrue(abs(batch_loss - exp_batch_loss) < tol)

            exp_avg_loss = total_nll / count
            self.assertTrue(abs(exp_avg_loss - bce.avg_loss) < tol)

        bce.reset()
        self.assertTrue(math.isnan(bce.avg_loss))

    def test_binary_cross_entropy_prob_weighted(self):

        tol = 1e-5
        
        prob = Variable(torch.rand(6).float())
        target = Variable(torch.FloatTensor([1, 0, 1, 0, 1, 0]))
        weight = torch.FloatTensor([.8, .2])

        bce = BinaryCrossEntropy(mode="prob", weight=weight)

        total_nll = 0
        count = 0
        for i in range(0, 6, 2):
            count += 2
            batch_nll = - weight[1] * torch.log(prob[i:i+1]).data[0]
            batch_nll -= weight[0] * torch.log(1 - prob[i+1:i+2]).data[0]
            total_nll += batch_nll
            exp_batch_loss = batch_nll / 2

            batch_loss = bce.eval(prob[i:i+2], target[i:i+2]).data[0]
            self.assertTrue(abs(batch_loss - exp_batch_loss) < tol)

            exp_avg_loss = total_nll / count
            self.assertTrue(abs(exp_avg_loss - bce.avg_loss) < tol)

        bce.reset()
        self.assertTrue(math.isnan(bce.avg_loss))

    def test_binary_cross_entropy_logit_weighted(self):

        tol = 1e-5
        
        logit = Variable(torch.rand(6).float() - .5)
        target = Variable(torch.FloatTensor([1, 0, 1, 0, 1, 0]))
        weight = torch.FloatTensor([.8, .2])

        bce = BinaryCrossEntropy(mode="logit", weight=weight)

        total_nll = 0
        count = 0
        for i in range(0, 6, 2):
            count += 2
            batch_nll = -weight[1] * torch.log(
                1 / (1 + torch.exp(-logit[i:i+1]))).data[0]
            batch_nll -= weight[0] * torch.log(
                1 - 1 / (1 + torch.exp(-logit[i+1:i+2]))).data[0]
            total_nll += batch_nll
            exp_batch_loss = batch_nll / 2

            batch_loss = bce.eval(logit[i:i+2], target[i:i+2]).data[0]
            self.assertTrue(abs(batch_loss - exp_batch_loss) < tol)

            exp_avg_loss = total_nll / count
            self.assertTrue(abs(exp_avg_loss - bce.avg_loss) < tol)

        bce.reset()
        self.assertTrue(math.isnan(bce.avg_loss))

    def test_binary_cross_entropy_logit(self):
        tol = 1e-5

        bce = BinaryCrossEntropy(mode="logit")
        logit = Variable(torch.rand(6).float() - .5)
        target = Variable(torch.FloatTensor([1, 0, 1, 0, 1, 0]))
        
        total_nll = 0
        count = 0
        for i in range(0, 6, 2):
            count += 2
            batch_nll = -torch.log(
                1 / (1 + torch.exp(-logit[i:i+1]))).data[0]
            batch_nll -= torch.log(
                1 - 1 / (1 + torch.exp(-logit[i+1:i+2]))).data[0]
            total_nll += batch_nll
            exp_batch_loss = batch_nll / 2
            batch_loss = bce.eval(logit[i:i+2], target[i:i+2]).data[0]
            self.assertTrue(abs(batch_loss - exp_batch_loss) < tol)

            exp_avg_loss = total_nll / count
            self.assertTrue(abs(exp_avg_loss - bce.avg_loss) < tol)

        bce.reset()
        self.assertTrue(math.isnan(bce.avg_loss))

    def test_binary_cross_entropy_prob(self):
        tol = 1e-5

        bce = BinaryCrossEntropy(mode="prob")
        
        prob = Variable(torch.rand(6).float())
        target = Variable(torch.FloatTensor([1, 0, 1, 0, 1, 0]))
        
        total_nll = 0
        count = 0
        for i in range(0, 6, 2):
            count += 2
            batch_nll = -torch.log(prob[i:i+1]).data[0]
            batch_nll -= torch.log(1 - prob[i+1:i+2]).data[0]
            total_nll += batch_nll
            exp_batch_loss = batch_nll / 2
            batch_loss = bce.eval(prob[i:i+2], target[i:i+2]).data[0]
            self.assertTrue(abs(batch_loss - exp_batch_loss) < tol)

            exp_avg_loss = total_nll / count
            self.assertTrue(abs(exp_avg_loss - bce.avg_loss) < tol)

        bce.reset()
        self.assertTrue(math.isnan(bce.avg_loss))

if __name__ == '__main__':
    unittest.main()
