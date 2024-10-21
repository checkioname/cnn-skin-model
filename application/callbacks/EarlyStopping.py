


class EarlyStopping():
    def __init__(self, patience, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0


    def __call__(self, current_score):
        if self.best_score is None:
            self.best_score = current_score
        elif current_score < self.best_score +self.min_delta:
            self.counter += 1
            if self.counter > self.patience:
                print("Encerrando o treinamento após {counter} épocas sem melhora")
                self.early_stop = True
        else: 
            self.best_score = current_score
            self.counter = 0


    