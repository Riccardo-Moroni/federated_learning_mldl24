import numpy as np

class ClientSelector:
    def __init__(self, params):
        K = params['K']
        C = params['C']
        
        self.S = np.array(range(K))
        self.sample_size = int(max(C*K, 1))
        
        self.participation = params['participation']
        if self.participation == 'uniform':
            self.p = None
        else:
            self.p = np.random.dirichlet(alpha=np.full(K, 1/params['gamma']))


    def sample(self):
        return np.random.choice(
                self.S, 
                self.sample_size, 
                p=self.p, 
                replace=False
            )
