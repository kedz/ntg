import unittest
import nt
import torch
import random
import math


class TestMultiClassCrossEntropy(unittest.TestCase):


    def test_cross_entropy_logit(self):
        tol = 1e-5
        num_points = 6
        num_classes = 3
        ce = nt.criterion.MultiClassCrossEntropy(mode="logit")
        logit = torch.rand(num_points, num_classes).float() - .5
        logit = torch.autograd.Variable(logit)
        target = torch.LongTensor([1, 0, 2, 2, 1, 0])
        random.shuffle(target)
        target = torch.autograd.Variable(target)

        total_nll = 0
        count = 0
        for i in range(0, 5, 2):
            count += 2
            
            batch_nll = 0
            for b in range(i, i + 2):
                t = target.data[b]
                ll = logit.data[b][t] \
                    - torch.log(torch.exp(logit[b]).sum()).data[0]
                batch_nll -= ll
            total_nll += batch_nll
            exp_batch_loss = batch_nll / 2
            
            batch_loss = ce.eval(logit[i:i+2], target[i:i+2]).data[0]
            self.assertTrue(abs(batch_loss - exp_batch_loss) < tol)

            exp_avg_loss = total_nll / count
            self.assertTrue(abs(exp_avg_loss - ce.avg_loss) < tol)

        ce.reset()
        self.assertTrue(math.isnan(ce.avg_loss))

    def test_cross_entropy_prob(self):
        tol = 1e-5
        num_points = 6
        num_classes = 3
        ce = nt.criterion.MultiClassCrossEntropy(mode="prob")
        logit = torch.rand(num_points, num_classes).float() - .5
        logit = torch.autograd.Variable(logit)
        prob = torch.nn.functional.softmax(logit)
        target = torch.LongTensor([1, 0, 2, 2, 1, 0])
        random.shuffle(target)
        target = torch.autograd.Variable(target)
        
        total_nll = 0
        count = 0
        for i in range(0, 5, 2):
            count += 2
            
            batch_nll = 0
            for b in range(i, i + 2):
                t = target.data[b]
                ll = torch.log(prob[b][t]).data[0]
                batch_nll -= ll
            total_nll += batch_nll
            exp_batch_loss = batch_nll / 2
            
            batch_loss = ce.eval(prob[i:i+2], target[i:i+2]).data[0]
            self.assertTrue(abs(batch_loss - exp_batch_loss) < tol)

            exp_avg_loss = total_nll / count
            self.assertTrue(abs(exp_avg_loss - ce.avg_loss) < tol)

        ce.reset()
        self.assertTrue(math.isnan(ce.avg_loss))


    def test_binary_cross_entropy_logit_weighted(self):

        tol = 1e-5
        
        num_points = 6
        num_classes = 3
        logit = torch.rand(num_points, num_classes).float() - .5
        logit = torch.autograd.Variable(logit)

        target = torch.autograd.Variable(torch.LongTensor([1, 2, 1, 0, 1, 0]))
        weight = torch.FloatTensor([.8, .2, .5])

        ce = nt.criterion.MultiClassCrossEntropy(mode="logit", weight=weight)

        total_nll = 0
        count = 0
        for i in range(0, 5, 2):
            count += 2
            
            batch_nll = 0
            for b in range(i, i + 2):

                t = target.data[b]

                ll = logit.data[b][t] \
                    - torch.log(torch.exp(logit[b]).sum()).data[0]
                batch_nll -= weight[t] * ll

            total_nll += batch_nll
            exp_batch_loss = batch_nll / 2

            batch_loss = ce.eval(logit[i:i+2], target[i:i+2]).data[0]
            self.assertTrue(abs(batch_loss - exp_batch_loss) < tol)

            exp_avg_loss = total_nll / count
            self.assertTrue(abs(exp_avg_loss - ce.avg_loss) < tol)

        ce.reset()
        self.assertTrue(math.isnan(ce.avg_loss))


    def test_binary_cross_entropy_prob_weighted(self):

        tol = 1e-5
        
        num_points = 6
        num_classes = 3
        logit = torch.rand(num_points, num_classes).float() - .5
        logit = torch.autograd.Variable(logit)
        prob = torch.nn.functional.softmax(logit)

        target = torch.autograd.Variable(torch.LongTensor([1, 2, 1, 0, 1, 0]))
        weight = torch.FloatTensor([.8, .2, .5])

        ce = nt.criterion.MultiClassCrossEntropy(mode="prob", weight=weight)

        total_nll = 0
        count = 0
        for i in range(0, 5, 2):
            count += 2
            
            batch_nll = 0
            for b in range(i, i + 2):

                t = target.data[b]

                ll = torch.log(prob[b][t]).data[0] 
                batch_nll -= weight[t] * ll

            total_nll += batch_nll
            exp_batch_loss = batch_nll / 2

            batch_loss = ce.eval(prob[i:i+2], target[i:i+2]).data[0]
            self.assertTrue(abs(batch_loss - exp_batch_loss) < tol)

            exp_avg_loss = total_nll / count
            self.assertTrue(abs(exp_avg_loss - ce.avg_loss) < tol)

        ce.reset()
        self.assertTrue(math.isnan(ce.avg_loss))

    def test_cross_entropy_logit_mask(self):
        tol = 1e-5
        num_points = 6
        num_classes = 3
        ce = nt.criterion.MultiClassCrossEntropy(mode="logit", mask_value=-1)
        logit = torch.rand(num_points, num_classes).float() - .5
        logit = torch.autograd.Variable(logit)
        target = torch.LongTensor([1, 0, -1, 2, -1, 0])
        random.shuffle(target)
        target = torch.autograd.Variable(target)

        
        total_nll = 0
        count = 0
        for i in range(0, 5, 2):
            
            batch_nll = 0
            batch_size = 0
            for b in range(i, i + 2):
                t = target.data[b]
                if t != -1:
                    count += 1
                    batch_size += 1
                else:
                    continue 
                ll = logit.data[b][t] \
                    - torch.log(torch.exp(logit[b]).sum()).data[0]
                batch_nll -= ll
            total_nll += batch_nll
            exp_batch_loss = batch_nll / batch_size
            
            batch_loss = ce.eval(logit[i:i+2], target[i:i+2]).data[0]
            self.assertTrue(abs(batch_loss - exp_batch_loss) < tol)

            exp_avg_loss = total_nll / count
            self.assertTrue(abs(exp_avg_loss - ce.avg_loss) < tol)

        ce.reset()
        self.assertTrue(math.isnan(ce.avg_loss))

    def test_cross_entropy_prob_mask(self):
        tol = 1e-5
        num_points = 6
        num_classes = 3
        ce = nt.criterion.MultiClassCrossEntropy(mode="prob", mask_value=-1)
        logit = torch.rand(num_points, num_classes).float() - .5
        logit = torch.autograd.Variable(logit)
        prob = torch.nn.functional.softmax(logit)
        target = torch.LongTensor([1, 0, -1, 2, -1, 0])
        random.shuffle(target)
        target = torch.autograd.Variable(target)

        
        total_nll = 0
        count = 0
        for i in range(0, 5, 2):
            
            batch_nll = 0
            batch_size = 0
            for b in range(i, i + 2):
                t = target.data[b]
                if t != -1:
                    count += 1
                    batch_size += 1
                else:
                    continue 
                ll = torch.log(prob[b][t]).data[0] 
                batch_nll -= ll
            total_nll += batch_nll
            exp_batch_loss = batch_nll / batch_size
            
            batch_loss = ce.eval(prob[i:i+2], target[i:i+2]).data[0]
            self.assertTrue(abs(batch_loss - exp_batch_loss) < tol)

            exp_avg_loss = total_nll / count
            self.assertTrue(abs(exp_avg_loss - ce.avg_loss) < tol)

        ce.reset()
        self.assertTrue(math.isnan(ce.avg_loss))

    def test_cross_entropy_logit_weight_mask(self):
        tol = 1e-5
        num_points = 6
        num_classes = 3
        logit = torch.rand(num_points, num_classes).float() - .5
        logit = torch.autograd.Variable(logit)
        target = torch.LongTensor([1, 0, -1, 2, -1, 0])
        random.shuffle(target)
        target = torch.autograd.Variable(target)
        weight = torch.FloatTensor([.8, .2, .5])
        ce = nt.criterion.MultiClassCrossEntropy(
            mode="logit", weight=weight, mask_value=-1)
        
        total_nll = 0
        count = 0
        for i in range(0, 5, 2):
            
            batch_nll = 0
            batch_size = 0
            for b in range(i, i + 2):
                t = target.data[b]
                if t != -1:
                    count += 1
                    batch_size += 1
                else:
                    continue 
                ll = logit.data[b][t] \
                    - torch.log(torch.exp(logit[b]).sum()).data[0]
                batch_nll -= weight[t] * ll
            total_nll += batch_nll
            exp_batch_loss = batch_nll / batch_size
            
            batch_loss = ce.eval(logit[i:i+2], target[i:i+2]).data[0]
            self.assertTrue(abs(batch_loss - exp_batch_loss) < tol)

            exp_avg_loss = total_nll / count
            self.assertTrue(abs(exp_avg_loss - ce.avg_loss) < tol)

        ce.reset()
        self.assertTrue(math.isnan(ce.avg_loss))

    def test_cross_entropy_prob_weight_mask(self):
        tol = 1e-5
        num_points = 6
        num_classes = 3
        logit = torch.rand(num_points, num_classes).float() - .5
        logit = torch.autograd.Variable(logit)
        prob = torch.nn.functional.softmax(logit)
        target = torch.LongTensor([1, 0, -1, 2, -1, 0])
        random.shuffle(target)
        target = torch.autograd.Variable(target)
        weight = torch.FloatTensor([.8, .2, .5])
        ce = nt.criterion.MultiClassCrossEntropy(
            mode="prob", weight=weight, mask_value=-1)
        
        total_nll = 0
        count = 0
        for i in range(0, 5, 2):
            
            batch_nll = 0
            batch_size = 0
            for b in range(i, i + 2):
                t = target.data[b]
                if t != -1:
                    count += 1
                    batch_size += 1
                else:
                    continue 
                ll = torch.log(prob[b][t]).data[0]
                batch_nll -= weight[t] * ll
            total_nll += batch_nll
            exp_batch_loss = batch_nll / batch_size
            
            batch_loss = ce.eval(prob[i:i+2], target[i:i+2]).data[0]
            self.assertTrue(abs(batch_loss - exp_batch_loss) < tol)

            exp_avg_loss = total_nll / count
            self.assertTrue(abs(exp_avg_loss - ce.avg_loss) < tol)

        ce.reset()
        self.assertTrue(math.isnan(ce.avg_loss))

if __name__ == '__main__':
    unittest.main()
