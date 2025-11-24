import torch.nn as nn

class Critic(nn.Module):
    def __init__(self, input_dim, hidden_dim=128):
        super(Critic, self).__init__()
        
        self.main = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(0.2), # Recomendado para o Crítico
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim * 2, 1) # Output: score escalar
        )

    def forward(self, x):
        return self.main(x)
class Generator(nn.Module):
    def __init__(self, z_dim, output_dim, hidden_dim=128):
        super(Generator, self).__init__()
        
        self.main = nn.Sequential(
            nn.Linear(z_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            # A função de ativação final (Tanh ou Sigmoid) depende da normalização dos seus dados
            nn.Linear(hidden_dim, output_dim),
            # nn.Tanh() # Se os dados forem normalizados para [-1, 1]
            # nn.Sigmoid() # Se os dados forem normalizados para [0, 1]
        )

    def forward(self, z):
        return self.main(z)
