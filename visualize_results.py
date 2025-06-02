import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

def create_metric_plots(summary_no_bootstrap_file='summary_no_bootstrap.csv', 
                       summary_with_bootstrap_file='summary_with_bootstrap.csv',
                       output_dir='.'):
    """
    Create individual and combined plots for accuracy metrics comparison
    
    Args:
        summary_no_bootstrap_file (str): Path to no bootstrap summary CSV file
        summary_with_bootstrap_file (str): Path to with bootstrap summary CSV file
        output_dir (str): Directory to save plots
    """
    
    # Read the summary data
    df_all_no_bootstrap = pd.read_csv(summary_no_bootstrap_file)
    df_all_with_bootstrap = pd.read_csv(summary_with_bootstrap_file)
    
    # Combine summaries for plotting
    df_combined = pd.concat([df_all_no_bootstrap, df_all_with_bootstrap])
    
    # Metrics to plot
    metrics = ['fdr', 'tpr', 'fpr', 'shd', 'nnz']
    
    # Create x-axis labels based on the actual weight ranges
    x_labels = [f"{idx+1}\n({-0.1-0.1*idx:.1f},{-1.0-0.1*idx:.1f})|({0.1+0.1*idx:.1f},{1.0+0.1*idx:.1f})" 
                for idx in range(10)]
    
    # Define explicit colors for consistency
    colors = ['#1f77b4', '#ff7f0e']  # Blue for False (No Bootstrap), Orange for True (With Bootstrap)
    hue_order = [False, True]
    method_labels = ['No Bootstrap', 'With Bootstrap']
    
    # Create individual plots for each metric
    print("Creating individual plots for each metric...")
    for metric in metrics:
        plt.figure(figsize=(8, 6))
        
        # Plot data
        ax = sns.lineplot(
            data=df_combined, 
            x='range_idx', 
            y=metric, 
            hue='bootstrap',
            marker='o',
            palette=colors,
            hue_order=hue_order
        )
        
        plt.title(f"{metric.upper()} by Weight Range")
        plt.xlabel("Weight Range Index")
        plt.ylabel(metric.upper())
        plt.grid(True, alpha=0.3)
        plt.xticks(range(10), x_labels, rotation=45)
        handles, _ = ax.get_legend_handles_labels()
        ax.legend(handles=handles, title='Method', labels=method_labels)
        plt.tight_layout()
        
        # Save individual plot
        plt.savefig(os.path.join(output_dir, f'accuracy_metrics_{metric}.png'), dpi=300)
        plt.savefig(os.path.join(output_dir, f'accuracy_metrics_{metric}.pdf'))
        plt.close()
        print(f"Saved plot for {metric}")
    
    # Create a combined plot as well
    print("Creating combined plot...")
    plt.figure(figsize=(15, 10))
    for i, metric in enumerate(metrics):
        plt.subplot(2, 3, i+1)
        
        ax = sns.lineplot(
            data=df_combined, 
            x='range_idx', 
            y=metric, 
            hue='bootstrap',
            marker='o',
            palette=colors,
            hue_order=hue_order
        )
        
        plt.title(f"{metric.upper()} by Weight Range")
        plt.xlabel("Weight Range Index")
        plt.ylabel(metric.upper())
        plt.grid(True, alpha=0.3)
        plt.xticks(range(10), [str(i+1) for i in range(10)])
        if i == 0:
            handles, _ = ax.get_legend_handles_labels()
            ax.legend(handles=handles, title='Method', labels=method_labels)
        else:
            ax.get_legend().remove()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'accuracy_metrics_comparison.png'), dpi=300)
    plt.savefig(os.path.join(output_dir, 'accuracy_metrics_comparison.pdf'))
    plt.close()
    print("Saved combined plot")
    
    print("\nVisualization complete!")
    print(f"Individual metric plots saved as 'accuracy_metrics_*.png/pdf'")
    print(f"Combined plot saved as 'accuracy_metrics_comparison.png/pdf'")


def print_summary_statistics(summary_no_bootstrap_file='summary_no_bootstrap.csv', 
                           summary_with_bootstrap_file='summary_with_bootstrap.csv'):
    """
    Print summary statistics for the results
    """
    df_no_bootstrap = pd.read_csv(summary_no_bootstrap_file)
    df_with_bootstrap = pd.read_csv(summary_with_bootstrap_file)
    
    metrics = ['fdr', 'tpr', 'fpr', 'shd', 'nnz']
    
    print("\n" + "="*60)
    print("SUMMARY STATISTICS")
    print("="*60)
    
    for metric in metrics:
        print(f"\n{metric.upper()}:")
        print(f"  No Bootstrap  - Mean: {df_no_bootstrap[metric].mean():.4f}, Std: {df_no_bootstrap[metric].std():.4f}")
        print(f"  With Bootstrap - Mean: {df_with_bootstrap[metric].mean():.4f}, Std: {df_with_bootstrap[metric].std():.4f}")
        
        # Calculate improvement
        improvement = df_with_bootstrap[metric].mean() - df_no_bootstrap[metric].mean()
        if metric in ['tpr']:  # Higher is better for TPR
            direction = "better" if improvement > 0 else "worse"
        elif metric in ['fdr', 'fpr', 'shd']:  # Lower is better for FDR, FPR, SHD
            direction = "better" if improvement < 0 else "worse"
        else:  # NNZ depends on context
            direction = "different"
        
        print(f"  Bootstrap is {direction} by {abs(improvement):.4f}")


if __name__ == "__main__":
    # Check if summary files exist
    if not os.path.exists('summary_no_bootstrap.csv'):
        print("Error: summary_no_bootstrap.csv not found!")
        print("Please run the main experiment first to generate summary files.")
        exit(1)
    
    if not os.path.exists('summary_with_bootstrap.csv'):
        print("Error: summary_with_bootstrap.csv not found!")
        print("Please run the main experiment first to generate summary files.")
        exit(1)
    
    # Create the plots
    create_metric_plots()
    
    # Print summary statistics
    print_summary_statistics()
