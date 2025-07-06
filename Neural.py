import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision.utils import save_image

# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Image size and transform
image_size = 356
loader = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor()
])

# Load image
def load_image(image_name):
    image = Image.open(image_name).convert("RGB")
    image = loader(image).unsqueeze(0)  # Add batch dimension
    return image.to(device)

# VGG Model definition
class VGG(nn.Module):
    def __init__(self):
        super(VGG, self).__init__()
        self.chosen_features = ['0', '5', '10', '19', '28']
        self.model = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).features[:29]

        for param in self.model.parameters():
            param.requires_grad = False

    def forward(self, x):
        features = []
        for layer_num, layer in enumerate(self.model):
            x = layer(x)
            if str(layer_num) in self.chosen_features:
                features.append(x)
        return features

# Load content and style images
original_img = load_image("content.jpg")
style_img = load_image("style.jpeg")

# Generated image starts as a copy of content
generated = original_img.clone().requires_grad_(True)

# Load model
model = VGG().to(device).eval()

# Hyperparameters
total_steps = 6000
learning_rate = 0.001
alpha = 1  # content weight
beta = 0.01  # style weight

# Optimizer
optimizer = optim.Adam([generated], lr=learning_rate)

# Training loop
for step in range(total_steps):
    generated_features = model(generated)
    original_features = model(original_img)
    style_features = model(style_img)

    style_loss = original_loss = 0

    for gen_feat, ori_feat, style_feat in zip(generated_features, original_features, style_features):
        batch_size, channel, height, width = gen_feat.shape

        # Content loss
        original_loss += torch.mean((gen_feat - ori_feat) ** 2)

        # Style loss via Gram matrix
        G = gen_feat.view(channel, height * width).mm(gen_feat.view(channel, height * width).t())
        A = style_feat.view(channel, height * width).mm(style_feat.view(channel, height * width).t())

        style_loss += torch.mean((G - A) ** 2)

    total_loss = alpha * original_loss + beta * style_loss

    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    if step % 200 == 0:
        print(f"Step [{step}/{total_steps}], Total Loss: {total_loss.item():.4f}")
        save_image(generated, "generated.jpg")
