import numpy as np

class LrScheduler():
    def __init__(self, patience, factor, optm):
        self.patience = patience
        self.patience_org = patience
        self.factor = factor
        self.optimizer = optm
        self.best_loss = np.inf


    def step(self, current_loss):
        if current_loss < self.best_loss:
            self.best_loss = current_loss
            self.patience = self.patience_org
            self.no_improvement_counter = 0
        else:
            self.patience -= 1            
            if self.patience < 1:
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] *= self.factor
          


    


     