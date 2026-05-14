# This is an modification of Adam optimization algorithm for minimizing a function f with its gradient df.
	
import time
import numpy as np

def f_optimize(f, df, x0, time_limit):
    """
    Optimizes a black-box function using a custom 'Tailwind' Adaptive Adam algorithm.
    """
    start_time = time.time()
    max_iterations = 10000
    grad_tol = 1e-6
    
    # Ensure inputs are numpy arrays for element-wise operations
    x = np.array(x0, dtype=float) 
    gradient = np.array(df(x), dtype=float)
    
    # ---------------------------------------------------------
    # Tracking the best solution
    # ---------------------------------------------------------
    best_x = x.copy()
    best_f = f(x)
    
    # --- Base Adam Hyperparameters ---
    base_alpha = 0.05      # Starting step size
    current_alpha = base_alpha 
    max_alpha = 0.5        # Speed limit to prevent the tailwind from blowing us off course
    
    beta1 = 0.9            # Decay rate for 1st moment (momentum)
    beta2 = 0.999          # Decay rate for 2nd moment (variance)
    epsilon = 1e-8         # Small constant to prevent division by zero
    
    # Initialize Moment Vectors
    m = np.zeros_like(x) 
    v = np.zeros_like(x) 
    
    iteration_count = 0
    
    while (time.time() - start_time < time_limit) and (iteration_count < max_iterations):
        
        # 1. Early stopping if we've reached a flat region (local minimum)
        if np.linalg.norm(gradient) < grad_tol:
            break
            
        iteration_count += 1
        
        # ---------------------------------------------------------
        # INNOVATION: "Tailwind" Adaptive Momentum
        # ---------------------------------------------------------
        # We calculate the dot product between our current gradient
        # and our accumulated momentum (m). 
        # 
        # - If > 0: "Tailwind". The gradient and momentum agree. 
        #   We are moving smoothly down a slope.
        # - If < 0: "Headwind". The gradient opposes momentum. 
        #   We have likely crossed a ravine and are heading uphill.
        # ---------------------------------------------------------
        
        alignment = np.sum(gradient * m)
        
        if alignment > 0:
            # TAILWIND: Confidently increase the step size by 5%, capped at max_alpha
            current_alpha = min(current_alpha * 1.05, max_alpha)
            
        elif alignment < 0:
            # HEADWIND: We overshot the minimum! 
            # 1. Kill the momentum instantly so we don't roll up the wrong side.
            m = np.zeros_like(x)
            # 2. Reset the learning rate to be cautious again.
            current_alpha = base_alpha
        
        
        # 2. Update biased first moment estimate (using potentially zeroed 'm')
        m = beta1 * m + (1 - beta1) * gradient
        
        # 3. Update biased second raw moment estimate (Variance tracking remains untouched)
        v = beta2 * v + (1 - beta2) * (gradient ** 2)
        
        # 4. Compute bias-corrected estimates
        m_hat = m / (1 - beta1 ** iteration_count)
        v_hat = v / (1 - beta2 ** iteration_count)
        
        # 5. Update the current guess using our dynamic 'current_alpha'
        x = x - current_alpha * (m_hat / (np.sqrt(v_hat) + epsilon))
        
        # ---------------------------------------------------------
        # Evaluate function to ensure we output the BEST approximate solution
        # ---------------------------------------------------------
        current_f = f(x)
        if current_f < best_f:
            best_f = current_f
            best_x = x.copy()
            
        # 6. Compute the gradient for the next iteration
        gradient = np.array(df(x), dtype=float)

    # Return the historical best, not necessarily the final iteration
    return best_x

# =================================================================
# 2. TEST BENCHMARKS
# =================================================================

def run_test(name, f, df, x0, expected_min_x, time_limit=5.0):
    print(f"--- Testing: {name} ---")
    print(f"Initial Guess: {x0}")
    
    start = time.time()
    x_opt = f_optimize(f, df, x0, time_limit)
    elapsed = time.time() - start
    
    print(f"Time Taken: {elapsed:.4f}s")
    print(f"Final Coordinate: {x_opt}")
    print(f"Final Objective Value f(x): {f(x_opt):.8e}")
    print(f"Target Minimum: 0.0")
    print("-" * 30 + "\n")

# A. Rosenbrock (Valley)
def f_rosen(x): return np.sum(100.0 * (x[1:] - x[:-1]**2)**2 + (1 - x[:-1])**2)
def df_rosen(x):
    grad = np.zeros_like(x)
    grad[:-1] += -400.0 * x[:-1] * (x[1:] - x[:-1]**2) - 2.0 * (1 - x[:-1])
    grad[1:] += 200.0 * (x[1:] - x[:-1]**2)
    return grad

# B. Rastrigin (Multimodal)
def f_rast(x): return 10 * len(x) + np.sum(x**2 - 10 * np.cos(2 * np.pi * x))
def df_rast(x): return 2.0 * x + 10.0 * 2.0 * np.pi * np.sin(2 * np.pi * x)

# C. Ill-Conditioned Quadratic
def f_ill(x): return 1e6 * x[0]**2 + x[1]**2
def df_ill(x): return np.array([2e6 * x[0], 2.0 * x[1]])

# =================================================================
# 3. EXECUTION
# =================================================================
if __name__ == "__main__":
    # Test 1: Rosenbrock (2D)
    run_test("Rosenbrock (The Valley)", f_rosen, df_rosen, np.array([-1.2, 1.0]), np.ones(2))
    
    # Test 2: Rastrigin (High Dimension - 5D)
    run_test("Rastrigin (Multimodal 5D)", f_rast, df_rast, np.array([2.5, -1.5, 3.0, -2.0, 0.5]), np.zeros(5))
    
    # Test 3: Ill-Conditioned (Extreme Scaling)
    run_test("Ill-Conditioned Quadratic", f_ill, df_ill, np.array([10.0, 10.0]), np.zeros(2))