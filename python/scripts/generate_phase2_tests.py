import torch
import argparse
import pandas as pd
from model import Generator

def generate_synthetic_tests(num_samples=500, checkpoint='models/generator_phase2_epoch50.pth', output_file='synthetic_tests_phase2.csv'):
    """Generate synthetic test data from trained generator"""
    
    device = torch.device('cpu')
    
    print(f"Using device: {device}")
    print(f"Loading generator from: {checkpoint}")
    
    # Load generator
    generator = Generator(100, 3)
    generator.load_state_dict(torch.load(checkpoint, map_location=device))
    generator.eval()
    
    # Generate samples
    print(f"Generating {num_samples} synthetic test samples...")
    with torch.no_grad():
        z = torch.randn(num_samples, 100, device=device)
        synthetic_data = generator(z).cpu().numpy()
    
    # Save to CSV
    df = pd.DataFrame(synthetic_data)
    df.to_csv(output_file, header=False, index=False)
    
    print(f"Synthetic tests saved to: {output_file}")
    print(f"   Shape: {synthetic_data.shape}")
    print(f"\nFirst 5 synthetic test samples:")
    print(df.head())

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate synthetic test data using trained WGAN-GP')
    parser.add_argument('--num-samples', type=int, default=500, help='Number of samples to generate')
    parser.add_argument('--checkpoint', type=str, default='models/generator_phase2_epoch50.pth', help='Path to generator checkpoint')
    parser.add_argument('--output-file', type=str, default='synthetic_tests_phase2.csv', help='Output CSV file path')
    
    args = parser.parse_args()
    
    generate_synthetic_tests(args.num_samples, args.checkpoint, args.output_file)
