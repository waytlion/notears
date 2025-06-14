import numpy as np
import scipy.linalg as slin
import scipy.optimize as sopt
from scipy.special import expit as sigmoid


def notears_linear(X, lambda1, loss_type, max_iter=100, h_tol=1e-8, rho_max=1e+16, w_threshold=0.1):
    """Solve min_W L(W; X) + lambda1 ‖W‖_1 s.t. h(W) = 0 using augmented Lagrangian.

    Args:
        X (np.ndarray): [n, d] sample matrix
        lambda1 (float): l1 penalty parameter
        loss_type (str): l2, logistic, poisson
        max_iter (int): max num of dual ascent steps
        h_tol (float): exit if |h(w_est)| <= htol
        rho_max (float): exit if rho >= rho_max
        w_threshold (float): drop edge if |weight| < threshold

    Returns:
        W_est (np.ndarray): [d, d] estimated DAG
    """
    def _loss(W):
        """Evaluate value and gradient of loss."""
        M = X @ W
        if loss_type == 'l2':
            R = X - M
            loss = 0.5 / X.shape[0] * (R ** 2).sum()
            G_loss = - 1.0 / X.shape[0] * X.T @ R
        elif loss_type == 'logistic':
            loss = 1.0 / X.shape[0] * (np.logaddexp(0, M) - X * M).sum()
            G_loss = 1.0 / X.shape[0] * X.T @ (sigmoid(M) - X)
        elif loss_type == 'poisson':
            S = np.exp(M)
            loss = 1.0 / X.shape[0] * (S - X * M).sum()
            G_loss = 1.0 / X.shape[0] * X.T @ (S - X)
        else:
            raise ValueError('unknown loss type')
        return loss, G_loss

    def _h(W):
        """Evaluate value and gradient of acyclicity constraint."""
        E = slin.expm(W * W)  # (Zheng et al. 2018)
        h = np.trace(E) - d
        #     # A different formulation, slightly faster at the cost of numerical stability
        #     M = np.eye(d) + W * W / d  # (Yu et al. 2019)
        #     E = np.linalg.matrix_power(M, d - 1)
        #     h = (E.T * M).sum() - d
        G_h = E.T * W * 2
        return h, G_h

    def _adj(w):
        """Convert doubled variables ([2 d^2] array) back to original variables ([d, d] matrix)."""
        return (w[:d * d] - w[d * d:]).reshape([d, d])

    def _func(w):
        """Evaluate value and gradient of augmented Lagrangian for doubled variables ([2 d^2] array)."""
        W = _adj(w)
        loss, G_loss = _loss(W)
        h, G_h = _h(W)
        obj = loss + 0.5 * rho * h * h + alpha * h + lambda1 * w.sum()
        G_smooth = G_loss + (rho * h + alpha) * G_h
        g_obj = np.concatenate((G_smooth + lambda1, - G_smooth + lambda1), axis=None)
        return obj, g_obj

    n, d = X.shape
    w_est, rho, alpha, h = np.zeros(2 * d * d), 1.0, 0.0, np.inf  # double w_est into (w_pos, w_neg)
    bnds = [(0, 0) if i == j else (0, None) for _ in range(2) for i in range(d) for j in range(d)]
    if loss_type == 'l2':
        X = X - np.mean(X, axis=0, keepdims=True)
    for _ in range(max_iter):
        w_new, h_new = None, None
        while rho < rho_max:
            sol = sopt.minimize(_func, w_est, method='L-BFGS-B', jac=True, bounds=bnds)
            w_new = sol.x
            h_new, _ = _h(_adj(w_new))
            if h_new > 0.25 * h:
                rho *= 10
            else:
                break
        w_est, h = w_new, h_new
        alpha += rho * h
        if h <= h_tol or rho >= rho_max:
            break
    W_est = _adj(w_est)
    W_est[np.abs(W_est) < w_threshold] = 0
    return W_est


if __name__ == '__main__':
    import utils
    import pandas as pd
    import os
    import matplotlib.pyplot as plt
    import seaborn as sns
    import time
    from datetime import timedelta

    n, d, s0, graph_type, sem_type = 100, 20, 20, 'ER', 'gauss'
    w_threshold = 0.1
    bootstrap_samples = 0
    n_seeds = 1
    n_W_Intervals=1
    # For tracking total execution time
    start_time_total = time.time()
    
    # Create folders for results
    detail_folder = 'detailed_results'
    os.makedirs(detail_folder, exist_ok=True)
    
    # Lists to store results for each weight range
    all_range_results_no_bootstrap = []
    all_range_results_with_bootstrap = []
    
    # Metrics to track
    metrics = ['fdr', 'tpr', 'fpr', 'shd', 'nnz']
    # Loop over 10 different weight ranges
    for range_idx in range(n_W_Intervals):
        # Start timing for this weight range
        start_time_range = time.time()
        
        # Adjust weight ranges based on iteration
        w_range_1 = (-0.1 - 0.1 * range_idx, -1.0 - 0.1 * range_idx)  # e.g., (-0.1, -1), (-0.2, -1.1), etc.
        w_range_2 = (0.1 +  0.1 * range_idx, 1.0 + 0.1 * range_idx)  # e.g., (0.1, 1), (0.1, 1.1), etc.
        
        print(f"\n===== Weight Range {range_idx+1} =====")
        print(f"w_range_1: {w_range_1}, w_range_2: {w_range_2}")
        
        # Lists to store results for this weight range
        results_no_bootstrap = []
        results_with_bootstrap = []
        
        # Run experiment for each seed
        for seed in range(n_seeds):
            print(f"Running seed {seed}")
            utils.set_random_seed(seed)
            
            # Generate data (seed-dependent)
            B_true = utils.simulate_dag(d, s0, graph_type)
            W_true = utils.simulate_parameter(B_true, (w_range_1, w_range_2))
            X = utils.simulate_linear_sem(W_true, n, sem_type)

            # Save data for this seed and range
            range_detail_folder = f'{detail_folder}/range_{range_idx+1}'
            os.makedirs(range_detail_folder, exist_ok=True)
            np.savetxt(f'{range_detail_folder}/W_true_seed_{seed}.csv', W_true, delimiter=',')
            np.savetxt(f'{range_detail_folder}/X_seed_{seed}.csv', X, delimiter=',')
            
            # Method 1: Direct estimation (no bootstrapping)
            W_est = notears_linear(X, lambda1=0.1, loss_type='l2')
            np.savetxt(f'{range_detail_folder}/W_est_no_bootstraped_seed_{seed}.csv', W_est, delimiter=',')
            acc_no_bootstrap = utils.count_accuracy(B_true, W_est != 0)
            
            # Store results with seed and range info
            result_no_bootstrap = {'seed': seed, 'range_idx': range_idx}
            result_no_bootstrap.update(acc_no_bootstrap)
            results_no_bootstrap.append(result_no_bootstrap)
            
            # Method 2: Bootstrap estimation
            W_est_bootstrapped = []
            for _ in range(bootstrap_samples):
                indices = np.random.choice(n, size=n, replace=True)
                subsample = X[indices]
                W_est_bootstrapped.append(notears_linear(subsample, lambda1=0.1, loss_type="l2"))
            
            # Stack and average
            W_stack = np.stack(W_est_bootstrapped)
            W_mean = np.mean(W_stack, axis=0)
            W_mean[np.abs(W_mean) < w_threshold] = 0
            
            # Evaluate bootstrapped model
            acc_with_bootstrap = utils.count_accuracy(B_true, W_mean != 0)
            
            # Store results with seed and range info
            result_with_bootstrap = {'seed': seed, 'range_idx': range_idx, 'n_bootstrap_samples': bootstrap_samples}
            result_with_bootstrap.update(acc_with_bootstrap)
            results_with_bootstrap.append(result_with_bootstrap)
            
            print(f"Seed {seed} - No bootstrap: {acc_no_bootstrap}")
            print(f"Seed {seed} - With bootstrap: {acc_with_bootstrap}")
        
        # Convert to DataFrames for this range
        df_no_bootstrap = pd.DataFrame(results_no_bootstrap)
        df_with_bootstrap = pd.DataFrame(results_with_bootstrap)
        
        # Save detailed results for this range
        df_no_bootstrap.to_csv(f'{detail_folder}/accuracies/accuracies_no_bootstrap_range_{range_idx+1}.csv', index=False)
        df_with_bootstrap.to_csv(f'{detail_folder}/accuracies/accuracies_with_bootstrap_range_{range_idx+1}.csv', index=False)
        
        # Calculate summary statistics for this range
        range_summary_no_bootstrap = df_no_bootstrap.describe().loc['mean', metrics].to_dict()
        range_summary_with_bootstrap = df_with_bootstrap.describe().loc['mean', metrics].to_dict()
        
        # Add range info to both summaries
        range_metadata = {
            'range_idx': range_idx,
            'w_range_1_low': w_range_1[0],
            'w_range_1_high': w_range_1[1],
            'w_range_2_low': w_range_2[0],
            'w_range_2_high': w_range_2[1]
        }

        range_summary_no_bootstrap.update(range_metadata)
        range_summary_no_bootstrap['bootstrap'] = False

        range_summary_with_bootstrap.update(range_metadata)
        range_summary_with_bootstrap['bootstrap'] = True
          # Store range summaries
        all_range_results_no_bootstrap.append(range_summary_no_bootstrap)
        all_range_results_with_bootstrap.append(range_summary_with_bootstrap)
        
        # Calculate and print the time taken for this weight range
        range_time = time.time() - start_time_range
        range_time_str = str(timedelta(seconds=int(range_time)))
        print(f"Range {range_idx+1} complete - results saved (Time taken: {range_time_str})")
    
    # Convert range summaries to DataFrames
    df_all_no_bootstrap = pd.DataFrame(all_range_results_no_bootstrap)
    df_all_with_bootstrap = pd.DataFrame(all_range_results_with_bootstrap)
      # Save summary results
    df_all_no_bootstrap.to_csv('summary_no_bootstrap.csv', index=False)
    df_all_with_bootstrap.to_csv('summary_with_bootstrap.csv', index=False)
    
    # Calculate total time
    total_time = time.time() - start_time_total
    total_time_str = str(timedelta(seconds=int(total_time)))
    
    # Print final summary
    print("\nExperiment complete!")
    print(f"Total execution time: {total_time_str}")    
    print("Summary files saved as 'summary_no_bootstrap.csv' and 'summary_with_bootstrap.csv'")
    print("Detailed results saved by weight range")
    print("\nTo create visualizations, run: python visualize_results.py")
