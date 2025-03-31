import torch
import numpy as np
import cv2


class GradCAM:
    def __init__(self, model, target_layer):
        """
        Inicializa o Grad-CAM.

        Args:
            model (torch.nn.Module): Modelo de rede neural (CNN ou Transformer).
            target_layer (torch.nn.Module): Camada alvo para extração de ativação e gradientes.
        """
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        # Hooks para capturar ativação e gradientes
        self.forward_hook = target_layer.register_forward_hook(self.save_activation)
        self.backward_hook = target_layer.register_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        """ Captura a ativação da camada alvo """
        self.activations = output.detach()

    def save_gradient(self, module, grad_input, grad_output):
        """ Captura os gradientes da camada alvo """
        self.gradients = grad_output[0].detach()

    def generate(self, input_tensor, class_idx=None):
        """
        Gera o mapa de Grad-CAM.

        Args:
            input_tensor (torch.Tensor): Imagem de entrada normalizada (B, C, H, W).
            class_idx (int, opcional): Índice da classe alvo. Se None, escolhe a classe de maior probabilidade.

        Returns:
            np.ndarray: Mapa de calor do Grad-CAM normalizado (H, W).
        """
        self.model.eval()

        # Forward pass
        output = self.model(input_tensor)
        if class_idx is None:
            class_idx = torch.argmax(output)

        # Backward pass para obter gradientes
        self.model.zero_grad()
        output[:, class_idx].backward()

        # Computar Grad-CAM
        gradients = self.gradients.mean(dim=[2, 3], keepdim=True)  # Média global
        cam = torch.sum(gradients * self.activations, dim=1)  # Somatório ponderado
        cam = torch.relu(cam)  # Aplicar ReLU

        # Normalizar para 0-1
        cam -= cam.min()
        cam /= cam.max()

        return cam.squeeze().cpu().numpy()

    def overlay_gradcam(self, img, heatmap, alpha=0.5):
        """Sobrepõe o mapa de calor Grad-CAM na imagem original."""
        heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))  # Redimensionar o mapa de calor
        heatmap = np.uint8(255 * heatmap)  # Converter para valores de 0 a 255
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)  # Aplicar colormap

        overlay = cv2.addWeighted(img, 1 - alpha, heatmap, alpha, 0)
        return overlay

    def remove_hooks(self):
        """ Remove os hooks registrados no modelo. """
        self.forward_hook.remove()
        self.backward_hook.remove()
