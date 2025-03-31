import torch 
from domain.Swim import SetupModelSwin
from torchvision import transforms
from PIL import Image

def plot_gradcam(model, img):
    model, _, _, _ = SetupModelSwin().setup_model()

def plot_gradcam(model, img_path, device='cpu'):
    model.eval()

    # Define as transformações para a imagem
    transform = transforms.Compose([
        transforms.CenterCrop(512),
        transforms.ToTensor(),
    ])

    # Carrega a imagem
    img_pil = Image.open(img_path).convert('RGB')
    img_tensor = transform(img_pil).unsqueeze(0).to(device)

    # Encontra a última camada convolucional (ou bloco similar) para o Grad-CAM
    target_layer = model.features[-1] # Experimente com outras camadas se esta não funcionar bem

    # Guarda os gradientes e as ativações da camada alvo
    feature_maps = []
    gradients = []
    def forward_hook(module, input, output):
        feature_maps.append(output.detach())

    def backward_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0].detach())

    hook_f = target_layer.register_forward_hook(forward_hook)
    hook_b = target_layer.register_backward_hook(backward_hook)

    # Realiza a inferência
    output = model(img_tensor)
    print("RESULTADO DA PREDICAO: ", output)
    _, predicted_class = torch.max(output, 1) # Para classificação multiclasse

    # Para classificação binária com Sigmoid
    predicted_probability = output.sigmoid().item()
    predicted_class_binary = 1 if predicted_probability > 0.5 else 0
    target_class = torch.tensor([predicted_class_binary]).to(device) # Define a classe alvo para o Grad-CAM

    # Calcula os gradientes para a classe alvo
    model.zero_grad()
    output.backward(gradient=torch.ones_like(output)) # Para classificação binária

    # Remove os hooks
    hook_f.remove()
    hook_b.remove()

    # Obtém os mapas de características e os gradientes
    pooled_gradients = torch.mean(gradients[0], dim=[0, 2, 3])
    feature_map = feature_maps[0].squeeze()

    # Pondera os mapas de características pelos gradientes
    for i in range(pooled_gradients.shape[0]):
        feature_map[i, :, :] *= pooled_gradients[i]

    # Cria o heatmap do Grad-CAM
    heatmap = torch.mean(feature_map, dim=0).cpu().numpy()
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)

    # Redimensiona o heatmap para o tamanho da imagem original
    img_cv = cv2.imread(img_path)
    heatmap = cv2.resize(heatmap, (img_cv.shape[1], img_cv.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    # Sobrepõe o heatmap na imagem original
    superimposed_img = cv2.addWeighted(img_cv, 0.6, heatmap, 0.4, 0)
    superimposed_img_rgb = cv2.cvtColor(superimposed_img, cv2.COLOR_BGR2RGB)

    # Exibe a imagem com o Grad-CAM
    plt.imshow(superimposed_img_rgb)
    plt.title(f'Predicted Probability: {predicted_probability:.4f}, Predicted Class: {predicted_class_binary}')
    plt.show()

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pretrained_file_path = '/home/king/Documents/PsoriasisEngineering/cnn-skin-model/model_engineering/runs/ml-model-test-1742432834.2114196/model.pt' 
    setup = SetupModelSwin()
    model, _, _, _ = setup.setup_model(device, pretrained_path=pretrained_file_path)


    image_derma = "/home/king/Documents/PsoriasisEngineering/cnn-skin-model/model_engineering/infrastructure/db/dermatite/COSTA SOARES, RHAYAN MIGUEL  (20240430090521504) 20240430091026337.jpg"
    image_path = '/home/king/Documents/PsoriasisEngineering/cnn-skin-model/data_engineering/database/psoriase_vulgar/MARIA ROSA DE JESUS SOUSA - 607682/DE JESUS SOUSA, MARIA ROSA  (20220714093119461) 20220714093322494.jpg'
    plot_gradcam(model, image_derma, device)