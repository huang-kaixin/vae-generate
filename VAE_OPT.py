import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

# 定义 VAE 编码器
class VAEEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, latent_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, latent_size * 2)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        mu, log_var = x.split(latent_size, dim=1)
        return mu, log_var

# 定义 VAE 解码器
class VAEDecoder(nn.Module):
    def __init__(self, latent_size, hidden_size, output_size):
        super().__init__()
        self.fc1 = nn.Linear(latent_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        return x


# 定义 VAE 模型
class VAE(nn.Module):
    def __init__(self, input_size, hidden_size, latent_size):
        super().__init__()
        self.encoder = VAEEncoder(input_size, hidden_size, latent_size)
        self.decoder = VAEDecoder(latent_size, hidden_size, input_size)

    def forward(self, x):
        mu, log_var = self.encoder(x)
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        z = mu + std * eps
        recon = self.decoder(z)
        return recon, mu, log_var

# 定义 VAE 损失函数
def vae_loss(recon, x, mu, log_var):
    recon_loss = nn.BCELoss(reduction='sum')(recon, x)
    kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    return recon_loss + kl_loss

# 加载 MNIST 数据集
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
mnist = datasets.MNIST(root='.', download=True, transform=transform)

# 定义训练参数
batch_size = 64
lr = 1e-3
num_epochs = 20

# 定义数据加载器
data_loader = DataLoader(mnist, batch_size=batch_size, shuffle=True) # shuffle=True 打乱数据

# 定义模型、优化器和损失函数
# 定义 VAE 模型
input_size = 28 * 28
hidden_size = 256
latent_size = 64
model = VAE(input_size, hidden_size, latent_size)

# 定义优化器
optimizer = optim.Adam(model.parameters(), lr=lr)

if __name__ == '__main__': # 仅在当前文件中运行时才执行以下代码
    # 训练 VAE 模型
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for x, _ in data_loader:
            x = x.view(-1, input_size)
            recon, mu, log_var = model(x)
            loss = vae_loss(recon, x, mu, log_var)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        # print(f'Epoch {epoch+1} loss: {epoch_loss / len(mnist):.4f}') 
        print(f'Epoch {epoch+1} loss: {loss.item():.3f}')             

    # 使用 VAE 生成图像
    with torch.no_grad():
        z = torch.randn(1, latent_size)
        image = model.decoder(z).view(28, 28)
        image = image.detach().numpy()
        plt.imshow(image, cmap='gray')
        plt.show() 

    # 保存模型
    torch.save(model.state_dict(), 'vae_opt.pth')

