import time
import numpy as np

def f_optimize(f, df, x0, time_limit):
    """
    Unconstrained optimization solver using the AdaBelief algorithm.
    AdaBelief is a variant of Adam that adapts the step size according to the 
    'belief' in the current gradient direction.
    """
    start_time = time.time()
    max_iterations = 50000
    grad_tol = 1e-6
    
    # Convert inputs to numpy arrays for element-wise vector operations
    x = np.array(x0, dtype=float) 
    gradient = np.array(df(x), dtype=float)
    iteration_count = 0
    
    # ==========================================
    # HYPERPARAMETERS
    # ==========================================
    # alpha is the learning rate. For black-box functions where the scale 
    # is unknown, 0.01 or 0.001 is often a safer starting point than 0.1.
    alpha = 0.01      
    beta1 = 0.9       # Exponential decay rate for the 1st moment (momentum)
    beta2 = 0.999     # Exponential decay rate for the 2nd moment (variance)
    epsilon = 1e-8    # Small constant to prevent division by zero
    
    # ==========================================
    # INITIALIZATION
    # ==========================================
    # m: 1st moment vector (tracks the moving average of the gradient)
    m = np.zeros_like(x) 
    
    # s: 2nd moment vector (tracks the variance of the prediction error in AdaBelief)
    # In standard Adam, this is usually called 'v' and tracks the squared gradient.
    s = np.zeros_like(x) 
    
    current_time = time.time()
    
    # Main Optimization Loop
    # Halts if time runs out OR max iterations are reached
    while ((current_time - start_time < time_limit) and (iteration_count < max_iterations)):
        
        # 1. Early stopping: If the gradient is practically zero, we are at a local minimum
        if np.linalg.norm(gradient) < grad_tol:
            break
            
        iteration_count += 1
        
        # ==========================================
        # STEP 2: STANDARD ADAM MECHANIC
        # ==========================================
        # Update the biased first moment estimate. 
        # This acts like momentum, smoothing out the gradient direction over time.
        m = beta1 * m + (1 - beta1) * gradient
        
        # ==========================================
        # STEP 3: ADABELIEF MODIFICATION (THE CORE INNOVATION)
        # ==========================================
        # STANDARD ADAM would do: v = beta2 * v + (1 - beta2) * (gradient ** 2)
        # ADABELIEF does: s = beta2 * s + (1 - beta2) * ((gradient - m) ** 2) + epsilon
        # 
        # Why? We calculate the squared difference between the observed gradient 
        # and the expected gradient (m). 
        # - If the gradient is exactly what we expected (gradient roughly equals m), 
        #   the error is 0, 's' stays small, and we take a LARGE step (high belief).
        # - If the gradient wildly contradicts what we expected, the error is large, 
        #   's' grows, and we take a SMALL step (low belief).
        # The extra + epsilon inside the update is recommended by the AdaBelief authors 
        # for numerical stability.
        s = beta2 * s + (1 - beta2) * ((gradient - m) ** 2) + epsilon
        
        # ==========================================
        # STEP 4: STANDARD ADAM MECHANIC
        # ==========================================
        # Bias correction for the first moment. 
        # Because 'm' was initialized with zeros, it is biased toward zero early on.
        # This division corrects that bias during the first few iterations.
        m_hat = m / (1 - beta1 ** iteration_count)
        
        # ==========================================
        # STEP 5: ADABELIEF MODIFICATION 
        # ==========================================
        # Bias correction for the variance vector 's'.
        # Similar to step 4, but applied to our new AdaBelief variance tracker.
        s_hat = s / (1 - beta2 ** iteration_count)
        
        # ==========================================
        # STEP 6: PARAMETER UPDATE
        # ==========================================
        # Update the current guess 'x'. 
        # We step in the direction of the momentum (m_hat), scaled inversely by 
        # the square root of our belief in that direction (s_hat).
        x = x - alpha * (m_hat / (np.sqrt(s_hat) + epsilon))
        
        # ==========================================
        # STEP 7: PREPARE FOR NEXT ITERATION
        # ==========================================
        # Query the black-box gradient function at the new position 'x'
        gradient = np.array(df(x), dtype=float)
        
        # Update the clock to ensure we don't breach the time_limit
        current_time = time.time()

    return x