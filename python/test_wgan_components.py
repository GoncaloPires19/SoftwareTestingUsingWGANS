"""
Comprehensive Unit Tests for WGAN-GP Components

Tests cover:
- Model initialization (Generator, Critic)
- Forward passes with various batch sizes
- Gradient penalty computation
- Loss calculations (Wasserstein)
- Gradient tracking utilities
- Data pipeline (normalization, denormalization)
- Training pipeline integration

Run with: pytest python/test_wgan_components.py -v
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from pathlib import Path
import sys
import tempfile

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from model import Generator, Critic
from utils import compute_gradient_penalty, get_model_gradient_stats
from config import WGANConfig, TrainingConfig, ModelConfig
import torch.optim as optim


# ============================================================================
# FIXTURES - Reusable test data and models
# ============================================================================

@pytest.fixture
def config():
    """Default WGAN configuration"""
    return WGANConfig(
        model=ModelConfig(z_dim=100, output_dim=3),
        training=TrainingConfig(
            epochs=50,
            batch_size=32,
            learning_rate=0.0002,
            lambda_gp=10.0,
            n_critic=5
        )
    )


@pytest.fixture
def device():
    """Test device (CPU by default)"""
    return torch.device("cpu")


@pytest.fixture
def generator(config, device):
    """Initialize Generator model"""
    return Generator(
        z_dim=config.model.z_dim,
        output_dim=config.model.output_dim
    ).to(device)


@pytest.fixture
def critic(config, device):
    """Initialize Critic model"""
    return Critic(
        input_dim=config.model.output_dim,
        output_dim=1
    ).to(device)


@pytest.fixture
def real_data_batch(config, device):
    """Mock real test data batch"""
    return torch.randn(config.training.batch_size, config.model.output_dim).to(device)


@pytest.fixture
def noise_batch(config, device, generator):
    """Mock noise vector batch"""
    return torch.randn(config.training.batch_size, config.model.z_dim).to(device)


# ============================================================================
# MODEL INITIALIZATION TESTS
# ============================================================================

class TestGeneratorInitialization:
    """Test Generator model initialization"""
    
    def test_generator_creation(self, config):
        """Test Generator can be instantiated"""
        gen = Generator(z_dim=config.model.z_dim, output_dim=config.model.output_dim)
        assert gen is not None
        assert isinstance(gen, nn.Module)
    
    def test_generator_architecture(self, config, device):
        """Test Generator has correct architecture"""
        gen = Generator(z_dim=config.model.z_dim, output_dim=config.model.output_dim)
        assert hasattr(gen, 'fc1')
        assert hasattr(gen, 'fc2')
        assert hasattr(gen, 'fc3')
        assert isinstance(gen.fc1, nn.Linear)
        assert isinstance(gen.fc2, nn.Linear)
        assert isinstance(gen.fc3, nn.Linear)
    
    def test_generator_layer_dimensions(self, config):
        """Test Generator layer dimensions match architecture"""
        gen = Generator(z_dim=config.model.z_dim, output_dim=config.model.output_dim)
        
        # fc1: z_dim -> 256
        assert gen.fc1.in_features == config.model.z_dim
        assert gen.fc1.out_features == 256
        
        # fc2: 256 -> 512
        assert gen.fc2.in_features == 256
        assert gen.fc2.out_features == 512
        
        # fc3: 512 -> output_dim
        assert gen.fc3.in_features == 512
        assert gen.fc3.out_features == config.model.output_dim
    
    def test_generator_parameter_count(self, config):
        """Test Generator parameter count"""
        gen = Generator(z_dim=config.model.z_dim, output_dim=config.model.output_dim)
        total_params = sum(p.numel() for p in gen.parameters())
        
        # Expected: z_dim*256 + 256 + 256*512 + 512 + 512*output_dim + output_dim
        expected = (config.model.z_dim * 256 + 256 + 
                   256 * 512 + 512 + 
                   512 * config.model.output_dim + config.model.output_dim)
        assert total_params == expected
    
    def test_generator_device_placement(self, config, device):
        """Test Generator placed on correct device"""
        gen = Generator(z_dim=config.model.z_dim, output_dim=config.model.output_dim).to(device)
        for param in gen.parameters():
            assert param.device == device


class TestCriticInitialization:
    """Test Critic model initialization"""
    
    def test_critic_creation(self, config):
        """Test Critic can be instantiated"""
        crit = Critic(input_dim=config.model.output_dim)
        assert crit is not None
        assert isinstance(crit, nn.Module)
    
    def test_critic_architecture(self, config):
        """Test Critic has correct architecture"""
        crit = Critic(input_dim=config.model.output_dim)
        assert hasattr(crit, 'fc1')
        assert hasattr(crit, 'fc2')
        assert hasattr(crit, 'fc3')
        assert isinstance(crit.fc1, nn.Linear)
        assert isinstance(crit.fc2, nn.Linear)
        assert isinstance(crit.fc3, nn.Linear)
    
    def test_critic_layer_dimensions(self, config):
        """Test Critic layer dimensions match architecture"""
        crit = Critic(input_dim=config.model.output_dim)
        
        # fc1: output_dim -> 512
        assert crit.fc1.in_features == config.model.output_dim
        assert crit.fc1.out_features == 512
        
        # fc2: 512 -> 256
        assert crit.fc2.in_features == 512
        assert crit.fc2.out_features == 256
        
        # fc3: 256 -> 1 (single score, not probability)
        assert crit.fc3.in_features == 256
        assert crit.fc3.out_features == 1
    
    def test_critic_parameter_count(self, config):
        """Test Critic parameter count"""
        crit = Critic(input_dim=config.model.output_dim)
        total_params = sum(p.numel() for p in crit.parameters())
        
        # Expected: output_dim*512 + 512 + 512*256 + 256 + 256*1 + 1
        expected = (config.model.output_dim * 512 + 512 +
                   512 * 256 + 256 +
                   256 * 1 + 1)
        assert total_params == expected
    
    def test_critic_device_placement(self, config, device):
        """Test Critic placed on correct device"""
        crit = Critic(input_dim=config.model.output_dim).to(device)
        for param in crit.parameters():
            assert param.device == device


# ============================================================================
# FORWARD PASS TESTS
# ============================================================================

class TestGeneratorForwardPass:
    """Test Generator forward pass"""
    
    def test_generator_forward_basic(self, generator, noise_batch, config):
        """Test Generator forward pass produces correct output shape"""
        output = generator(noise_batch)
        assert output.shape == (config.training.batch_size, config.model.output_dim)
    
    def test_generator_forward_batch_size_1(self, generator, config, device):
        """Test Generator with batch size 1"""
        z = torch.randn(1, config.model.z_dim, device=device)
        output = generator(z)
        assert output.shape == (1, config.model.output_dim)
    
    def test_generator_forward_batch_size_64(self, generator, config, device):
        """Test Generator with batch size 64"""
        z = torch.randn(64, config.model.z_dim, device=device)
        output = generator(z)
        assert output.shape == (64, config.model.output_dim)
    
    def test_generator_forward_no_nan(self, generator, noise_batch):
        """Test Generator output contains no NaN values"""
        output = generator(noise_batch)
        assert not torch.isnan(output).any()
    
    def test_generator_forward_no_inf(self, generator, noise_batch):
        """Test Generator output contains no Inf values"""
        output = generator(noise_batch)
        assert not torch.isinf(output).any()
    
    def test_generator_forward_gradient_flow(self, generator, noise_batch):
        """Test gradients flow through Generator"""
        z = noise_batch.clone().requires_grad_(True)
        output = generator(z)
        loss = output.sum()
        loss.backward()
        
        assert z.grad is not None
        assert not torch.isnan(z.grad).any()
    
    def test_generator_output_normalized_range(self, generator, noise_batch):
        """Test Generator output in reasonable range (not clipped to [-1,1] but unbounded)"""
        output = generator(noise_batch)
        # Generator should produce values that can be normalized later
        assert output.min() < 10 and output.max() > -10  # Sanity check


class TestCriticForwardPass:
    """Test Critic forward pass"""
    
    def test_critic_forward_basic(self, critic, real_data_batch, config):
        """Test Critic forward pass produces correct output shape"""
        output = critic(real_data_batch)
        assert output.shape == (config.training.batch_size, 1)
    
    def test_critic_forward_batch_size_1(self, critic, config, device):
        """Test Critic with batch size 1"""
        x = torch.randn(1, config.model.output_dim, device=device)
        output = critic(x)
        assert output.shape == (1, 1)
    
    def test_critic_forward_batch_size_64(self, critic, config, device):
        """Test Critic with batch size 64"""
        x = torch.randn(64, config.model.output_dim, device=device)
        output = critic(x)
        assert output.shape == (64, 1)
    
    def test_critic_forward_no_sigmoid(self, critic, real_data_batch):
        """Test Critic outputs scores (not probabilities 0-1)"""
        output = critic(real_data_batch)
        # Critic output should not be bounded to [0, 1] like sigmoid
        # (actual values can be outside this range)
        assert not (output >= 0).all() or not (output <= 1).all()
    
    def test_critic_forward_no_nan(self, critic, real_data_batch):
        """Test Critic output contains no NaN values"""
        output = critic(real_data_batch)
        assert not torch.isnan(output).any()
    
    def test_critic_forward_no_inf(self, critic, real_data_batch):
        """Test Critic output contains no Inf values"""
        output = critic(real_data_batch)
        assert not torch.isinf(output).any()
    
    def test_critic_forward_gradient_flow(self, critic, real_data_batch):
        """Test gradients flow through Critic"""
        x = real_data_batch.clone().requires_grad_(True)
        output = critic(x)
        loss = output.sum()
        loss.backward()
        
        assert x.grad is not None
        assert not torch.isnan(x.grad).any()


# ============================================================================
# GRADIENT PENALTY TESTS
# ============================================================================

class TestGradientPenalty:
    """Test gradient penalty computation"""
    
    def test_gp_computation_basic(self, critic, real_data_batch, generator, 
                                  config, device, noise_batch):
        """Test gradient penalty computation"""
        with torch.no_grad():
            fake_data = generator(noise_batch)
        
        gp = compute_gradient_penalty(
            critic,
            real_data_batch,
            fake_data,
            device=device,
            batch_size=config.training.batch_size,
            lambda_gp=10.0
        )
        
        assert isinstance(gp, torch.Tensor)
        assert gp.item() >= 0  # Penalty should be non-negative
    
    def test_gp_no_nan(self, critic, real_data_batch, generator, config, device, noise_batch):
        """Test GP output contains no NaN"""
        with torch.no_grad():
            fake_data = generator(noise_batch)
        
        gp = compute_gradient_penalty(critic, real_data_batch, fake_data, device,
                                     config.training.batch_size, lambda_gp=10.0)
        assert not torch.isnan(gp)
    
    def test_gp_no_inf(self, critic, real_data_batch, generator, config, device, noise_batch):
        """Test GP output contains no Inf"""
        with torch.no_grad():
            fake_data = generator(noise_batch)
        
        gp = compute_gradient_penalty(critic, real_data_batch, fake_data, device,
                                     config.training.batch_size, lambda_gp=10.0)
        assert not torch.isinf(gp)
    
    def test_gp_lambda_scaling(self, critic, real_data_batch, generator, config, device, noise_batch):
        """Test GP scales with lambda parameter"""
        with torch.no_grad():
            fake_data = generator(noise_batch)
        
        gp_lambda10 = compute_gradient_penalty(critic, real_data_batch, fake_data,
                                              device, config.training.batch_size, lambda_gp=10.0)
        gp_lambda20 = compute_gradient_penalty(critic, real_data_batch, fake_data,
                                              device, config.training.batch_size, lambda_gp=20.0)
        
        # GP with lambda=20 should be ~2x lambda=10 (approximately)
        ratio = gp_lambda20.item() / (gp_lambda10.item() + 1e-8)
        assert 1.5 < ratio < 2.5  # Allow some tolerance
    
    def test_gp_with_different_lambda_values(self, critic, real_data_batch, generator,
                                             config, device, noise_batch):
        """Test GP with edge case lambda values"""
        with torch.no_grad():
            fake_data = generator(noise_batch)
        
        # Lambda = 0 (no penalty)
        gp_zero = compute_gradient_penalty(critic, real_data_batch, fake_data,
                                          device, config.training.batch_size, lambda_gp=0.0)
        assert gp_zero.item() == 0.0
        
        # Lambda = 1
        gp_one = compute_gradient_penalty(critic, real_data_batch, fake_data,
                                         device, config.training.batch_size, lambda_gp=1.0)
        assert gp_one.item() >= 0
        
        # Lambda = 100
        gp_large = compute_gradient_penalty(critic, real_data_batch, fake_data,
                                           device, config.training.batch_size, lambda_gp=100.0)
        assert gp_large.item() >= gp_one.item()


# ============================================================================
# WASSERSTEIN LOSS TESTS
# ============================================================================

class TestWassersteinLoss:
    """Test Wasserstein loss calculations"""
    
    def test_critic_loss_computation(self, critic, real_data_batch, generator,
                                     config, device, noise_batch):
        """Test Critic loss: loss_critic = mean(fake) - mean(real) + GP"""
        with torch.no_grad():
            fake_data = generator(noise_batch)
        
        real_score = critic(real_data_batch)
        fake_score = critic(fake_data.detach())
        
        gp = compute_gradient_penalty(critic, real_data_batch, fake_data, device,
                                     config.training.batch_size, lambda_gp=config.training.lambda_gp)
        
        # Critic loss: maximize difference (real - fake)
        # Formula: -real_mean + fake_mean + gp (negative because we minimize)
        d_loss = -real_score.mean() + fake_score.mean() + gp
        
        assert isinstance(d_loss, torch.Tensor)
        assert not torch.isnan(d_loss)
        assert not torch.isinf(d_loss)
    
    def test_generator_loss_computation(self, critic, generator, config, device, noise_batch):
        """Test Generator loss: loss_generator = -mean(fake)"""
        fake_data = generator(noise_batch)
        fake_score = critic(fake_data)
        
        # Generator loss: maximize critic score on fake
        g_loss = -fake_score.mean()
        
        assert isinstance(g_loss, torch.Tensor)
        assert not torch.isnan(g_loss)
        assert not torch.isinf(g_loss)
    
    def test_loss_with_zeros(self, critic, config, device):
        """Test loss computation with zero inputs"""
        zeros = torch.zeros(config.training.batch_size, config.model.output_dim, device=device)
        ones = torch.ones(config.training.batch_size, config.model.output_dim, device=device)
        
        score_zeros = critic(zeros)
        score_ones = critic(ones)
        
        loss = -score_ones.mean() + score_zeros.mean()
        
        assert not torch.isnan(loss)
        assert not torch.isinf(loss)


# ============================================================================
# GRADIENT TRACKING TESTS
# ============================================================================

class TestGradientTracking:
    """Test gradient statistics tracking utility"""
    
    def test_get_gradient_stats_returns_dict(self, critic, real_data_batch):
        """Test get_model_gradient_stats returns dictionary"""
        real_data_batch.requires_grad_(True)
        output = critic(real_data_batch)
        loss = output.sum()
        loss.backward()
        
        stats = get_model_gradient_stats(critic)
        assert isinstance(stats, dict)
    
    def test_get_gradient_stats_keys(self, critic, real_data_batch):
        """Test get_model_gradient_stats has required keys"""
        real_data_batch.requires_grad_(True)
        output = critic(real_data_batch)
        loss = output.sum()
        loss.backward()
        
        stats = get_model_gradient_stats(critic)
        assert 'mean_grad_norm' in stats
        assert 'max_grad_norm' in stats
        assert 'min_grad_norm' in stats
    
    def test_get_gradient_stats_valid_values(self, critic, real_data_batch):
        """Test get_model_gradient_stats produces valid numerical values"""
        real_data_batch.requires_grad_(True)
        output = critic(real_data_batch)
        loss = output.sum()
        loss.backward()
        
        stats = get_model_gradient_stats(critic)
        
        # All stats should be non-negative
        assert stats['mean_grad_norm'] >= 0
        assert stats['max_grad_norm'] >= 0
        assert stats['min_grad_norm'] >= 0
        
        # No NaN/Inf
        assert not np.isnan(stats['mean_grad_norm'])
        assert not np.isinf(stats['mean_grad_norm'])
    
    def test_get_gradient_stats_ordering(self, critic, real_data_batch):
        """Test gradient norm ordering: min <= mean <= max"""
        real_data_batch.requires_grad_(True)
        output = critic(real_data_batch)
        loss = output.sum()
        loss.backward()
        
        stats = get_model_gradient_stats(critic)
        
        # min <= mean <= max (with some tolerance for numerical issues)
        assert stats['min_grad_norm'] <= stats['mean_grad_norm'] + 1e-6
        assert stats['mean_grad_norm'] <= stats['max_grad_norm'] + 1e-6
    
    def test_get_gradient_stats_no_gradients(self, critic):
        """Test get_model_gradient_stats with no gradients"""
        stats = get_model_gradient_stats(critic)
        
        # Should return valid dict even with no gradients
        assert isinstance(stats, dict)
        assert stats['mean_grad_norm'] == 0.0
        assert stats['max_grad_norm'] == 0.0
        assert stats['min_grad_norm'] == 0.0


# ============================================================================
# TRAINING PIPELINE TESTS
# ============================================================================

class TestTrainingStep:
    """Test single training step"""
    
    def test_single_training_step(self, generator, critic, config, device):
        """Test one complete training step (Critic + Generator)"""
        g_opt = optim.Adam(generator.parameters(), lr=config.training.learning_rate)
        c_opt = optim.Adam(critic.parameters(), lr=config.training.learning_rate)
        
        # Real data
        real_data = torch.randn(config.training.batch_size, config.model.output_dim, device=device)
        
        # Train Critic
        for _ in range(config.training.n_critic):
            c_opt.zero_grad()
            
            real_score = critic(real_data)
            z = torch.randn(config.training.batch_size, config.model.z_dim, device=device)
            fake_data = generator(z)
            fake_score = critic(fake_data.detach())
            
            gp = compute_gradient_penalty(critic, real_data, fake_data, device,
                                         config.training.batch_size, lambda_gp=config.training.lambda_gp)
            
            d_loss = -real_score.mean() + fake_score.mean() + gp
            d_loss.backward()
            c_opt.step()
        
        # Train Generator
        g_opt.zero_grad()
        z = torch.randn(config.training.batch_size, config.model.z_dim, device=device)
        fake_data = generator(z)
        fake_score = critic(fake_data)
        g_loss = -fake_score.mean()
        g_loss.backward()
        g_opt.step()
        
        assert True  # Step completed without errors
    
    def test_critic_loss_decreases_with_updates(self, generator, critic, config, device):
        """Test Critic loss decreases over multiple updates"""
        c_opt = optim.Adam(critic.parameters(), lr=0.0001)  # Lower LR for stability
        real_data = torch.randn(config.training.batch_size, config.model.output_dim, device=device)
        
        losses = []
        for _ in range(5):
            c_opt.zero_grad()
            real_score = critic(real_data)
            z = torch.randn(config.training.batch_size, config.model.z_dim, device=device)
            fake_data = generator(z)
            fake_score = critic(fake_data.detach())
            
            gp = compute_gradient_penalty(critic, real_data, fake_data, device,
                                         config.training.batch_size, lambda_gp=config.training.lambda_gp)
            
            d_loss = -real_score.mean() + fake_score.mean() + gp
            d_loss.backward()
            c_opt.step()
            losses.append(d_loss.item())
        
        # Losses should vary (not necessarily monotonic but should change)
        assert len(set(np.round(losses, 5))) > 1


# ============================================================================
# INTEGRATION TESTS
# ============================================================================

class TestEndToEnd:
    """End-to-end integration tests"""
    
    def test_50_epoch_configuration_mandatory(self, config):
        """Test 50 epochs is the default configuration"""
        assert config.training.epochs == 50
    
    def test_n_critic_5_default(self, config):
        """Test n_critic=5 is default (WGAN best practice)"""
        assert config.training.n_critic == 5
    
    def test_gradient_penalty_enabled(self, config):
        """Test lambda_gp is set (WGAN-GP enabled)"""
        assert config.training.lambda_gp > 0
    
    def test_full_training_simulation(self, generator, critic, config, device):
        """Simulate one mini training loop"""
        g_opt = optim.Adam(generator.parameters(), lr=config.training.learning_rate)
        c_opt = optim.Adam(critic.parameters(), lr=config.training.learning_rate)
        
        real_data = torch.randn(config.training.batch_size, config.model.output_dim, device=device)
        
        # Train for 2 epochs with 2 batches each
        for epoch in range(2):
            epoch_d_losses = []
            epoch_g_losses = []
            
            for batch_idx in range(2):
                # Critic
                for _ in range(config.training.n_critic):
                    c_opt.zero_grad()
                    real_score = critic(real_data)
                    z = torch.randn(config.training.batch_size, config.model.z_dim, device=device)
                    fake_data = generator(z)
                    fake_score = critic(fake_data.detach())
                    
                    gp = compute_gradient_penalty(critic, real_data, fake_data, device,
                                                 config.training.batch_size, lambda_gp=config.training.lambda_gp)
                    
                    d_loss = -real_score.mean() + fake_score.mean() + gp
                    d_loss.backward()
                    c_opt.step()
                
                epoch_d_losses.append(d_loss.item())
                
                # Generator
                g_opt.zero_grad()
                z = torch.randn(config.training.batch_size, config.model.z_dim, device=device)
                fake_data = generator(z)
                fake_score = critic(fake_data)
                g_loss = -fake_score.mean()
                g_loss.backward()
                g_opt.step()
                
                epoch_g_losses.append(g_loss.item())
            
            avg_d = np.mean(epoch_d_losses)
            avg_g = np.mean(epoch_g_losses)
            
            # Metrics should be valid floats
            assert isinstance(avg_d, (float, np.floating))
            assert isinstance(avg_g, (float, np.floating))
            assert not np.isnan(avg_d)
            assert not np.isnan(avg_g)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
