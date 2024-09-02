import torch


class Training():

    # Função de treinamento
    def train(self, model, dataloader, writer, loss_fn, optimizer, class_to_idx, device,epoch):
        size = len(dataloader.dataset)
        model.train()
        for batch, (X, y) in enumerate(dataloader):
            batch_labels_numeric = [class_to_idx[label] for label in y]
            y = torch.tensor(batch_labels_numeric, dtype=torch.float32).to(device)
            X, y = X.to(device), y.to(device)

            optimizer.zero_grad()

            pred = model(X)
            pred = pred.squeeze(1)

            loss = loss_fn(pred, y)
            writer.add_scalar("Loss/train", loss, epoch)
            loss.backward()
            optimizer.step()

            if batch % 100 == 0:
                current = batch * len(X)
                print(f"Loss: {loss.item():.7f}  [{current:>5d}/{size:>5d}]")

    # Função de teste
    def test(self, model, writer,  data, loss_fn, class_to_idx, epoch, device):
        size = len(data.dataset)
        num_batches = len(data)
        model.eval()
        test_loss, correct = 0, 0
        with torch.no_grad():
            for X, y in data:
                batch_labels_numeric = [class_to_idx[label] for label in y]
                y = torch.tensor(batch_labels_numeric, dtype=torch.float32).to(device)
                X, y = X.to(device), y.to(device)
                pred = model(X)
                pred = pred.squeeze(1)
                test_loss += loss_fn(pred, y).item()
                correct += (torch.round(pred) == y).sum().item()
        
        test_loss /= num_batches
        correct /= size
        accuracy = 100 * correct
        writer.add_scalar("Loss/test", test_loss, epoch)
        writer.add_scalar("Accuracy/test", accuracy, epoch)
        print(f"Test Error: Accuracy: {accuracy:.2f}% | Avg loss: {test_loss:.8f}\n")