import torch


class Training():
    def __init__(self, dataloader,model, writer) -> None:
        self.dataloader = dataloader
        self.model = model
        self.writer = writer

    # Função de treinamento
    def train(self, loss_fn, optimizer, class_to_idx, device,epoch):
        size = len(self.dataloader.dataset)
        self.model.train()
        for batch, (X, y) in enumerate(self.dataloader):
            batch_labels_numeric = [class_to_idx[label] for label in y]
            y = torch.tensor(batch_labels_numeric, dtype=torch.float32).to(device)
            X, y = X.to(device), y.to(device)

            optimizer.zero_grad()

            pred = self.model(X)
            pred = pred.squeeze(1)

            loss = loss_fn(pred, y)
            self.writer.add_scalar("Loss/train", loss, epoch)
            loss.backward()
            optimizer.step()

            if batch % 100 == 0:
                current = batch * len(X)
                print(f"Loss: {loss.item():.7f}  [{current:>5d}/{size:>5d}]")

    # Função de teste
    def test(self, loss_fn, class_to_idx, epoch, device):
        size = len(self.dataloader.dataset)
        num_batches = len(self.dataloader)
        self.model.eval()
        test_loss, correct = 0, 0
        with torch.no_grad():
            for X, y in self.dataloader:
                batch_labels_numeric = [class_to_idx[label] for label in y]
                y = torch.tensor(batch_labels_numeric, dtype=torch.float32).to(device)
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
        print(f"Test Error: Accuracy: {accuracy:.2f}% | Avg loss: {test_loss:.8f}\n")
