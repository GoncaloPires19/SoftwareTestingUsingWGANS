import torch

def compute_gradient_penalty(critic, real_samples, fake_samples, device):
    # 1. Amostragem de pontos intermediários (mistura de real e fake)
    alpha = torch.rand(real_samples.size(0), 1, device=device)
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    
    # 2. Obter a saída do Crítico para os pontos intermediários
    d_interpolates = critic(interpolates)
    
    # 3. Calcular os gradientes (o coração do WGAN-GP)
    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=torch.ones(d_interpolates.size(), device=device),
        create_graph=True, # Necessário para o otimizador do Gerador
        retain_graph=True,
        only_inputs=True,
    )[0]
    
    # 4. Calcular a norma L2 dos gradientes e a Penalidade (Lambda=10 é comum)
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    
    return gradient_penalty