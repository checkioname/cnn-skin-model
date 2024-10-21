import torch

class Training():
    def __init__(self, trainloader, testloader,model, writer) -> None:
        self.trainloader = trainloader
        self.testloader = testloader

        self.model = model
        self.writer = writer

    # Função de treinamento
    def train(self, loss_fn, optimizer, device,epoch):
        size = len(self.trainloader.dataset)
        self.model.train()
        for batch, (X, y) in enumerate(self.trainloader):
            X, y = X.to(device), y.to(device)

            optimizer.zero_grad()

            pred = self.model(X)
            pred = pred.squeeze(1)
            loss = loss_fn(pred, y)

            self.writer.add_scalar("Loss/train", loss, epoch)
            
            loss.backward()
            # gradient clipping - evita que gradiente muito grande exploda (pode retornar null no backward)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            optimizer.step()
            if batch % 100 == 0:
                current = batch * len(X)
                print(f"Loss: {loss.item():.7f}  [{current:>5d}/{size:>5d}]")

    # Função de teste
    def test(self, loss_fn, epoch, device):
        size = len(self.testloader.dataset)
        num_batches = len(self.testloader)
        self.model.eval()
        test_loss, correct = 0, 0
        with torch.no_grad():
            for X, y in self.testloader:
                X, y = X.to(device), y.to(device)
                pred = self.model(X)
                pred = pred.squeeze(1)
                test_loss += loss_fn(pred, y).item()
                correct += (torch.round(pred) == y).sum().item()
        
        test_loss /= num_batches
        correct /= size
        accuracy = 100 * correct
        self.writer.add_scalar("Loss/test", test_loss, epoch)
        self.writer.add_scalar("Accuracy/test", accuracy, epoch)
        print(f"Test:\n Accuracy: {accuracy:.2f}% | Avg loss: {test_loss:.8f}\n")
