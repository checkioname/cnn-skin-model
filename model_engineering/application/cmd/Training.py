import torch
from application.callbacks import EarlyStopping
from torchmetrics import Accuracy, Precision, Recall, F1Score
from torch.cuda.amp import autocast, GradScaler

class Training():
    def __init__(self, trainloader, testloader,model, writer, callbacks) -> None:
        self.trainloader = trainloader
        self.testloader = testloader

        self.model = model
        self.writer = writer
        self.should_stop = False
        self.callbacks = callbacks or []

    # Função de treinamento
    def train(self, loss_fn, optimizer, device,epoch):
        size = len(self.trainloader.dataset)
        self.model.train()
        # scaler = GradScaler()
        for batch, (X, y) in enumerate(self.trainloader):
            X, y = X.to(device), y.to(device)

            optimizer.zero_grad()
            # with autocast():
            pred = self.model(X)
            pred = pred.squeeze(1)
            loss = loss_fn(pred, y)

            self.writer.add_scalar("Loss/train", loss, epoch)

            # scaler.scale(loss).backward()
            loss.backward()
            # gradient clipping - evita que gradiente muito grande exploda (pode retornar null no backward)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            # scaler.step(optimizer)
            # scaler.update()
            optimizer.step()
            if batch % 100 == 0:
                current = batch * len(X)
                print(f"Loss: {loss.item():.7f}  [{current:>5d}/{size:>5d}]")


        for callback in self.callbacks: 
            callback.on_epoch(epoch, self.model, loss)
            if isinstance(callback, EarlyStopping):
                if callback.early_stopping is True:
                    self.should_stop = True

                

    # Função de teste
    def test(self, loss_fn, epoch, device):
        accuracy_metric = Accuracy(task="binary").to(device)
        precision_metric = Precision(task="binary").to(device)
        recall_metric = Recall(task="binary").to(device)
        f1_metric = F1Score(task="binary").to(device)

        num_batches = len(self.testloader)
        self.model.eval()
        test_loss = 0 
        with torch.no_grad():
            for X, y in self.testloader:
                X, y = X.to(device), y.to(device)
                pred = self.model(X)
                pred = pred.squeeze(1)

                test_loss += loss_fn(pred, y).item()
                
                accuracy_metric.update(pred, y)
                precision_metric.update(pred, y)
                recall_metric.update(pred, y)
                f1_metric.update(pred, y)
        
        #perda média
        test_loss /= num_batches
        
        # Calculando os valores finais das métricas
        accuracy = accuracy_metric.compute().item() * 100
        precision = precision_metric.compute().item()
        recall = recall_metric.compute().item()
        f1_score = f1_metric.compute().item()
        
        self.writer.add_scalar("Loss/test", test_loss, epoch)
        self.writer.add_scalar("Accuracy/test", accuracy, epoch)
        self.writer.add_scalar("Precision/test", precision, epoch)
        self.writer.add_scalar("Recall/test", recall, epoch)
        self.writer.add_scalar("F1Score/test", f1_score, epoch)
        print(f"Test:\n Accuracy: {accuracy:.2f}% | Avg loss: {test_loss:.8f}\n")

        # Zerando as métricas para o próximo ciclo de avaliação
        accuracy_metric.reset()
        precision_metric.reset()
        recall_metric.reset()
        f1_metric.reset()
        return test_loss
