import numpy
import torch


class p_r():
    def __init__(self):
        self.TP = numpy.zeros((4,), dtype=int)
        self.TN = numpy.zeros((4,), dtype=int)
        self.FP = numpy.zeros((4,), dtype=int)
        self.FN = numpy.zeros((4,), dtype=int)

    def one_batch(self, pr, la):
        pr = pr.cpu().numpy()
        la = la.cpu().numpy()
        for i in range(pr.shape[0]):
            if pr[i] == la[i]:
                self.TP[pr[i]] += 1
                self.TN = self.TN + numpy.ones((4,), dtype=int)
                self.TN[pr[i]] -= 1
            else:
                self.FP[pr[i]] += 1
                self.FN[la[i]] += 1
                self.TN = self.TN + numpy.ones((4,), dtype=int)
                self.TN[pr[i]] -= 1
                self.TN[la[i]] -= 1

    def result(self):
        precise = numpy.zeros((4,),dtype=float)
        recall = numpy.zeros((4,),dtype=float)
        for i in range(4):
            precise[i]=self.TP[i]/(self.TP[i]+self.FP[i])
            recall[i]=self.TP[i]/(self.TP[i]+self.FN[i])
        return precise,recall