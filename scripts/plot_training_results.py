"""
Standalone script to generate training visualization plots from saved model checkpoints.

Reads trainer_state.json from checkpoint directories and generates:
1. Training Loss over Steps
2. Learning Rate Schedule
3. Training Progress

Usage:
    python scripts/plot_training_results.py --output_dir out/bert_cs
"""

import json
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np


def load_training_history(output_dir):
    """Load training history from trainer_state.json in checkpoint directories."""
    output_path = Path(output_dir)
    
    # Look for trainer_state.json in checkpoint subdirectories
    checkpoints = sorted(output_path.glob('checkpoint-*/trainer_state.json'))
    
    if not checkpoints:
        # Try root directory
        root_state = output_path / 'trainer_state.json'
        if root_state.exists():
            checkpoints = [root_state]
    
    if not checkpoints:
        raise FileNotFoundError(f"No trainer_state.json found in {output_dir} or its checkpoints")
    
    # Use the most recent checkpoint
    trainer_state_path = checkpoints[-1]
    print(f"Loading training history from: {trainer_state_path}")
    
    with open(trainer_state_path, 'r') as f:
        state = json.load(f)
    
    return state


def extract_metrics(log_history):
    """Extract all available metrics from log history."""
    # Collect all data points
    steps = []
    epochs = []
    train_loss = []
    learning_rates = []
    grad_norms = []
    
    for entry in log_history:
        # Training loss entries
        if 'loss' in entry:
            if 'step' in entry:
                steps.append(entry['step'])
            if 'epoch' in entry:
                epochs.append(entry['epoch'])
                train_loss.append(entry['loss'])
            if 'learning_rate' in entry:
                learning_rates.append(entry['learning_rate'])
            if 'grad_norm' in entry:
                grad_norms.append(entry['grad_norm'])
    
    print(f"\nExtracted data:")
    print(f"  - Training loss points: {len(train_loss)}")
    print(f"  - Epochs: {epochs if epochs else 'None'}")
    print(f"  - Learning rates: {len(learning_rates)}")
    print(f"  - Gradient norms: {len(grad_norms)}")
    
    return {
        'steps': steps,
        'epochs': epochs,
        'train_loss': train_loss,
        'learning_rates': learning_rates,
        'grad_norms': grad_norms
    }


def load_test_metrics(output_dir):
    """Load test metrics from test_metrics.json if it exists."""
    test_metrics_path = Path(output_dir) / 'test_metrics.json'
    
    if test_metrics_path.exists():
        with open(test_metrics_path, 'r') as f:
            return json.load(f)
    return None


def plot_all_metrics(metrics, test_metrics, output_dir, total_epochs):
    """Generate training visualization plots."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Create figure with subplots
    n_plots = 3 if metrics['learning_rates'] else 2
    fig = plt.figure(figsize=(16, 5 * ((n_plots + 1) // 2)))
    
    plot_idx = 1
    
    # Plot 1: Training Loss vs Epoch
    ax1 = plt.subplot(2, 2, plot_idx)
    if metrics['train_loss'] and metrics['epochs']:
        ax1.plot(metrics['epochs'], metrics['train_loss'], 'o-', 
                label='Training Loss', linewidth=2.5, markersize=8, color='#1f77b4')
        ax1.set_xlabel('Epoch', fontsize=13, fontweight='bold')
        ax1.set_ylabel('Loss', fontsize=13, fontweight='bold')
        ax1.set_title('Training Loss over Epochs', fontsize=15, fontweight='bold', pad=15)
        ax1.legend(fontsize=11, framealpha=0.9)
        ax1.grid(True, alpha=0.3, linestyle='--')
        ax1.tick_params(labelsize=11)
    plot_idx += 1
    
    # Plot 2: Learning Rate Schedule
    if metrics['learning_rates'] and metrics['epochs']:
        ax2 = plt.subplot(2, 2, plot_idx)
        ax2.plot(metrics['epochs'], metrics['learning_rates'], 's-', 
                label='Learning Rate', linewidth=2.5, markersize=8, color='#ff7f0e')
        ax2.set_xlabel('Epoch', fontsize=13, fontweight='bold')
        ax2.set_ylabel('Learning Rate', fontsize=13, fontweight='bold')
        ax2.set_title('Learning Rate Schedule', fontsize=15, fontweight='bold', pad=15)
        ax2.legend(fontsize=11, framealpha=0.9)
        ax2.grid(True, alpha=0.3, linestyle='--')
        ax2.tick_params(labelsize=11)
        ax2.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))
        plot_idx += 1
    
    # Plot 3: Test Metrics (if available)
    if test_metrics:
        ax3 = plt.subplot(2, 2, plot_idx)
        
        metrics_to_plot = ['precision', 'recall', 'f1', 'accuracy']
        values = [test_metrics.get(m, 0) for m in metrics_to_plot]
        colors = ['#2ca02c', '#d62728', '#9467bd', '#8c564b']
        
        bars = ax3.bar(metrics_to_plot, values, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
        
        # Add value labels on bars
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height,
                    f'{val:.4f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        ax3.set_ylabel('Score', fontsize=13, fontweight='bold')
        ax3.set_title('Final Test Metrics', fontsize=15, fontweight='bold', pad=15)
        ax3.set_ylim([0, 1.1])
        ax3.grid(True, alpha=0.3, axis='y', linestyle='--')
        ax3.tick_params(labelsize=11)
        plot_idx += 1
    
    # Plot 4: Gradient Norms (if available)
    if metrics['grad_norms'] and metrics['epochs']:
        ax4 = plt.subplot(2, 2, plot_idx)
        ax4.plot(metrics['epochs'], metrics['grad_norms'], '^-', 
                label='Gradient Norm', linewidth=2.5, markersize=8, color='#e377c2')
        ax4.set_xlabel('Epoch', fontsize=13, fontweight='bold')
        ax4.set_ylabel('Gradient Norm', fontsize=13, fontweight='bold')
        ax4.set_title('Gradient Norm over Epochs', fontsize=15, fontweight='bold', pad=15)
        ax4.legend(fontsize=11, framealpha=0.9)
        ax4.grid(True, alpha=0.3, linestyle='--')
        ax4.tick_params(labelsize=11)
    
    plt.tight_layout(pad=3.0)
    
    # Save combined figure
    combined_path = output_path / 'training_metrics_combined.png'
    plt.savefig(combined_path, dpi=300, bbox_inches='tight')
    print(f"✓ Combined training metrics saved to: {combined_path}")
    plt.close()
    
    # Save individual loss plot
    save_loss_plot(metrics, output_path)
    
    # Save test metrics plot if available
    if test_metrics:
        save_test_metrics_plot(test_metrics, output_path)


def save_loss_plot(metrics, output_path):
    """Save standalone loss plot."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    if metrics['train_loss'] and metrics['epochs']:
        ax.plot(metrics['epochs'], metrics['train_loss'], 'o-', 
               label='Training Loss', linewidth=2.5, markersize=8, color='#1f77b4')
        
        ax.set_xlabel('Epoch', fontsize=13, fontweight='bold')
        ax.set_ylabel('Loss', fontsize=13, fontweight='bold')
        ax.set_title('Training Loss Convergence', fontsize=15, fontweight='bold')
        ax.legend(fontsize=12)
        ax.grid(True, alpha=0.3)
        
        # Add annotation for final loss
        final_loss = metrics['train_loss'][-1]
        final_epoch = metrics['epochs'][-1]
        ax.annotate(f'Final Loss: {final_loss:.4f}',
                   xy=(final_epoch, final_loss),
                   xytext=(10, 10), textcoords='offset points',
                   fontsize=11, fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7),
                   arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
    
    plt.tight_layout()
    loss_path = output_path / 'training_loss.png'
    plt.savefig(loss_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Training loss plot saved to: {loss_path}")


def save_test_metrics_plot(test_metrics, output_path):
    """Save standalone test metrics plot."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    metrics_to_plot = ['precision', 'recall', 'f1', 'accuracy']
    labels = ['Precision', 'Recall', 'F1 Score', 'Accuracy']
    values = [test_metrics.get(m, 0) for m in metrics_to_plot]
    colors = ['#2ca02c', '#d62728', '#9467bd', '#8c564b']
    
    bars = ax.bar(labels, values, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
    
    # Add value labels on bars
    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{val:.4f}', ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    ax.set_ylabel('Score', fontsize=13, fontweight='bold')
    ax.set_title('XLM-RoBERTa Token Classification: Test Performance', fontsize=15, fontweight='bold')
    ax.set_ylim([0, 1.1])
    ax.grid(True, alpha=0.3, axis='y', linestyle='--')
    ax.tick_params(labelsize=11)
    
    plt.tight_layout()
    metrics_path = output_path / 'test_metrics_bar.png'
    plt.savefig(metrics_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Test metrics bar chart saved to: {metrics_path}")


def main():
    parser = argparse.ArgumentParser(description='Generate training visualization plots')
    parser.add_argument('--output_dir', type=str, default='out/bert_cs',
                       help='Directory containing model checkpoints')
    args = parser.parse_args()
    
    print(f"\n{'='*60}")
    print(f"Generating Training Visualization Plots")
    print(f"{'='*60}\n")
    
    try:
        # Load training history
        state = load_training_history(args.output_dir)
        log_history = state.get('log_history', [])
        print(f"Found {len(log_history)} log entries")
        
        # Get total epochs from state
        total_epochs = state.get('epoch', 3)
        print(f"Total training epochs: {total_epochs}")
        
        # Extract metrics
        metrics = extract_metrics(log_history)
        
        # Load test metrics if available
        test_metrics = load_test_metrics(args.output_dir)
        if test_metrics:
            print(f"\nTest metrics found:")
            for k, v in test_metrics.items():
                print(f"  - {k}: {v:.4f}")
        
        # Generate plots
        plot_all_metrics(metrics, test_metrics, args.output_dir, total_epochs)
        
        print(f"\n{'='*60}")
        print(f"✓ All plots generated successfully!")
        print(f"{'='*60}\n")
        
    except Exception as e:
        print(f"\n❌ Error: {e}\n")
        import traceback
        traceback.print_exc()
        raise


if __name__ == '__main__':
    main()