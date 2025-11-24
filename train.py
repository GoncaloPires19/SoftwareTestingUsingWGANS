# train.py

# Importações Essenciais do PyTorch
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Importações para Manipulação de Dados 
import pandas as pd
import numpy as np


from .model import Generator, Critic 
from .utils import compute_gradient_penalty 

# --- 1. CONFIGURAÇÃO DE HIPERPARÂMETROS E AMBIENTE ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LEARNING_RATE = 1e-4  # Taxa de aprendizado comum para WGAN-GP
BATCH_SIZE = 64
Z_DIM = 100           # Dimensão do vetor de ruído (z)
INPUT_DIM = 25        # <--- IMPORTANTE: Ajuste este valor (dimensão dos seus dados tabulares)
NUM_EPOCHS = 10
CRITIC_ITERATIONS = 5 # n_critic
LAMBDA_GP = 10        # lambda_gp

# --- 2. PREPARAÇÃO DO DATA LOADER (Simulação de Dados de Teste Tabulares) ---

# Implemente aqui o carregamento real dos seus dados pré-processados
# (e.g., carregar de 'data/processed/dados_processados.pt')
def get_data_loader(input_dim, batch_size, num_samples=1000):
    # SIMULAÇÃO: Criação de dados tabulares aleatórios
    # Na implementação real, os dados serão preparados aqui.
    data = torch.randn(num_samples, input_dim) 
    dataset = TensorDataset(data)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    print(f"Dados simulados carregados: {num_samples} amostras, batch_size={batch_size}")
    return loader

data_loader = get_data_loader(INPUT_DIM, BATCH_SIZE)

# --- 3. INICIALIZAÇÃO DAS REDES E OTIMIZADORES ---
critic = Critic(INPUT_DIM).to(DEVICE)
generator = Generator(Z_DIM, INPUT_DIM).to(DEVICE)

# Otimizadores com learning rates definidas e betas ajustados
optimizer_D = optim.Adam(critic.parameters(), lr=LEARNING_RATE, betas=(0.0, 0.9))
optimizer_G = optim.Adam(generator.parameters(), lr=LEARNING_RATE, betas=(0.0, 0.9))

# --- 4. CICLO DE TREINO PRINCIPAL ---
print(f"Iniciando treino na Epoch 1 de {NUM_EPOCHS} no dispositivo: {DEVICE}")

for epoch in range(NUM_EPOCHS):
    for batch_idx, data in enumerate(data_loader):
        
        # O DataLoader retorna uma tupla. Extrair apenas o tensor de dados.
        real_data = data[0].to(DEVICE)
        
        # --- TREINO DO CRÍTICO (MAXIMIZAR a perda de Wasserstein) ---
        for _ in range(CRITIC_ITERATIONS):
            # 1. Zero os gradientes do Crítico
            optimizer_D.zero_grad()
            
            # 2. Dados Falsos (Generated)
            noise = torch.randn(BATCH_SIZE, Z_DIM, device=DEVICE)
            # USAR A INSTÂNCIA 'generator', NÃO A CLASSE 'Generator'
            fake_data = generator(noise).detach() 
            
            # 3. Calcular a Perda (Score)
            d_real = critic(real_data)
            d_fake = critic(fake_data)
            
            # 4. Penalidade de Gradiente (WGAN-GP)
            gp = compute_gradient_penalty(critic, real_data, fake_data, DEVICE)
            
            # 5. Perda Total do Crítico (Wasserstein Distance + Penalty)
            # Loss = Score_Fake - Score_Real + lambda * GP
            loss_d = d_fake.mean() - d_real.mean() + LAMBDA_GP * gp
            
            # 6. Atualizar pesos do Crítico
            loss_d.backward()
            optimizer_D.step()

        # --- TREINO DO GERADOR (MINIMIZAR a perda) ---
        
        # Zero os gradientes do Gerador
        optimizer_G.zero_grad()
        
        # Dados Falsos (Generated)
        noise = torch.randn(BATCH_SIZE, Z_DIM, device=DEVICE)
        # USAR A INSTÂNCIA 'generator'
        fake_data = generator(noise) 
        
        # Calcular Score Falso
        # USAR A INSTÂNCIA 'critic'
        d_fake = critic(fake_data)
        
        # Loss do Gerador (Loss = -Crítico_Score_Falso.mean(). Minimizar -> Maximizar Score)
        loss_g = -d_fake.mean()
        
        # Atualizar pesos do Gerador
        loss_g.backward()
        optimizer_G.step()
    
    # Imprimir métricas a cada Epoch
    if epoch % 1 == 0:
        print(f"Epoch [{epoch}/{NUM_EPOCHS}] Loss D: {loss_d.item():.4f}, Loss G: {loss_g.item():.4f}")
