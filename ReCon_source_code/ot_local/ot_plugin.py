import torch
from ot_local.ot_pytorch_sinkhorn import sinkhorn, OptimalTransportLayer


class OTPlugin:
    def __init__(self, user_num, item_num, sinkhorn_gamma, sinkhorn_maxiter, iterative_optimization=False):
        super().__init__()
        self.user_num = user_num
        self.item_num = item_num
        self.sinkhorn_gamma = sinkhorn_gamma
        self.sinkhorn_maxiter = sinkhorn_maxiter
        self.iterative_optimization = iterative_optimization
        self.ot_l = OptimalTransportLayer(gamma=self.sinkhorn_gamma, eps=1.0e-6, maxiters=self.sinkhorn_maxiter, logspace=False, method='approx')
        
    def get_all_user_item_list(self):
        users_all = torch.LongTensor([int(u / self.item_num) for u in range(0, self.user_num * self.item_num)])
        items_all = torch.LongTensor([int(i % self.item_num) for i in range(0, self.user_num * self.item_num)])
        if torch.cuda.is_available():
            users_all = users_all.cuda()
            items_all = items_all.cuda()
        return users_all, items_all

    def get_sinkhorn_loss(self, predictions, shape=None):
        if shape is None:
            shape = (self.user_num, self.item_num)

        P = predictions.reshape(shape)
        M = torch.log((1-P)/P)
        

        # a = sinkhorn(M.unsqueeze(0).detach().clone(), gamma=self.sinkhorn_gamma, maxiters=self.sinkhorn_maxiter)
        a = sinkhorn(M.unsqueeze(0), gamma=self.sinkhorn_gamma, maxiters=self.sinkhorn_maxiter)
        # a = self.ot_l(M.unsqueeze(0))
        sinkhorn_loss = torch.dot(a.flatten(),M.unsqueeze(0).flatten())        
        D = -torch.log(1-predictions)
        sinkhorn_loss += D.sum()
        
        return sinkhorn_loss

