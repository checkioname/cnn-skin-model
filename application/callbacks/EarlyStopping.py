


class EarlyStopping():
    def __init__(self, patience, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False


    def __call__(self, current_loss):
        if self.best_loss is None:
            self.best_loss = current_loss
        elif current_loss > self.best_loss +self.min_delta:
            self.counter += 1
            if self.counter > self.patience:
                print("Encerrando o treinamento após {counter} épocas sem melhora")
                self.early_stop = True
        else: 
            self.best_loss = current_loss
            self.counter = 0


    