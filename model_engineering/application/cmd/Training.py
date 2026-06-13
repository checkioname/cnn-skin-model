import torch
from application.callbacks import EarlyStopping
from torchmetrics import Accuracy, Precision, Recall, F1Score
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image


TARGET_LAYERS = {
    'vgg16':     lambda m: [m.features[-1]],
    'resnet152': lambda m: [m.layer4[-1]],
    'vit':       lambda m: [m.encoder.layers[-1].ln_1],
    'swin':      lambda m: [m.features[-1][-1].norm1],
}


class Training():
    def __init__(self, trainloader, testloader, model, writer, callbacks,
                 model_name=None, cam_sample=None) -> None:
        self.trainloader = trainloader
        self.testloader = testloader

        self.model = model
        self.writer = writer
        self.should_stop = False
        self.callbacks = callbacks or []
        self.cam_sample = cam_sample


    # Função de treinamento
    def train(self, loss_fn, optimizer, device, epoch):
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
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            optimizer.step()
            if batch % 100 == 0:
                current = batch * len(X)
                print(f"Loss: {loss.item():.7f}  [{current:>5d}/{size:>5d}]")


        for callback in self.callbacks:
            callback.on_epoch(epoch, self.model, loss)

    def test(self, loss_fn, epoch, device):
        accuracy_metric = Accuracy(task="binary").to(device)
        precision_metric = Precision(task="binary").to(device)
        recall_metric = Recall(task="binary").to(device)
        f1_metric = F1Score(task="binary").to(device)

        num_batches = len(self.testloader)
        self.model.eval()
        test_loss = 0 
        with torch.no_grad():
            for i, (X, y) in enumerate(self.testloader):
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

            accuracy_metric.reset()
            precision_metric.reset()
            recall_metric.reset()
            f1_metric.reset()
            return test_loss

    def visualize_gradcam(self, epoch, device):
        if self.cam_sample is None:
            return

        raw_img, input_tensor = self.cam_sample
        model_name = self.model.__class__.__name__.lower()

        target_layers = None
        for key, fn in TARGET_LAYERS.items():
            if key in model_name:
                target_layers = fn(self.model)
                break
        if target_layers is None:
            return

        self.model.eval()
        input_tensor = input_tensor.to(device)

        with GradCAM(model=self.model, target_layers=target_layers) as cam:
            grayscale_cam = cam(input_tensor=input_tensor)[0, :]
            visualization = show_cam_on_image(raw_img, grayscale_cam, use_rgb=True)

        visualization_tensor = torch.from_numpy(visualization).permute(2, 0, 1).float() / 255
        self.writer.add_image('GradCAM/epoch', visualization_tensor, epoch)
