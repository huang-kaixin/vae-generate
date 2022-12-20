import torch
import matplotlib.pyplot as plt
# from VAE_OPT import model, input_size, mnist # 从 VAE.py 中导入模型、输入大小和 MNIST 数据集，选择已训练的模型，VAE.py 或 VAE_OPT.py
from VAE import model, input_size, mnist

# 调用模型
model.load_state_dict(torch.load('vae.pth')) # 选择加载的模型，vae.pth 或 vae_opt.pth

# 定义数字 6 的样本图像
sample_image = mnist[7][0] # [9][0]:4, [7][0]:3, [0][0]:5

# 使用 VAE 的编码器将样本图像编码为 latent variables
mu, log_var = model.encoder(sample_image.view(-1, input_size)) # view(-1, input_size):将图像转换为 1*784 的张量

# 将生成的 latent variables 作为输入传递给 VAE 的解码器，生成数字图像
generated_image = model.decoder(mu).view(28, 28)

# 显示原始图像和生成的图像
plt.subplot(1, 2, 1)
plt.title('Original Image')
plt.imshow(sample_image.view(28, 28), cmap='gray') # view(28, 28):将图像转换为 28*28 的张量 cmap='gray':灰度图
plt.subplot(1, 2, 2)
plt.title('Generated Image')
plt.imshow(generated_image.detach().numpy(), cmap='gray')
plt.show()
