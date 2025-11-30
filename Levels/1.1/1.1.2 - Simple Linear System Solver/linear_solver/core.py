"""
Core functions for the Simple Linear System Solver.
"""

def gaussian_elimination(A, b):
    """
    Performs Gaussian elimination to convert the augmented matrix [A|b] 
    to row echelon form.
    
    Args:
        A (list of lists): The coefficient matrix.
        b (list): The constant vector.
        
    Returns:
        tuple: (A_ref, b_ref) where A_ref is the matrix in row echelon form
               and b_ref is the transformed constant vector.
    """
    def vec_mul(c, v):
        return [c * vi for vi in v]
    
    def vec_sub(v1, v2):
        return [v1i - v2i for v1i, v2i in zip(v1, v2)]

    A_ref = A.copy()
    b_ref = b.copy()
    n = len(b)
    for k in range(n):
        for i in range(k+1, n):
            l = A_ref[i][k]/A_ref[k][k]
            A_ref[i] = vec_sub(A_ref[i], vec_mul(l, A_ref[k]))
            b_ref[i] = b_ref[i] - l * b_ref[k]
    return (A_ref, b_ref)


def back_substitution(A, b):
    """
    Solves for x in the system Ax = b where A is in row echelon form
    using back substitution.
    
    Args:
        A (list of lists): The coefficient matrix in row echelon form.
        b (list): The constant vector.
        
    Returns:
        list: The solution vector x.
    """
    n = len(b)
    x = [0] * n
    for k in reversed(range(n)):
        x[k] = (b[k] - sum(x[i] * A[k][i] for i in range(k+1, n))) / A[k][k]

    return (x)


def solve_linear_system(A, b):
    """
    Solves the linear system Ax = b using Gaussian elimination and back substitution.
    
    Args:
        A (list of lists): The coefficient matrix.
        b (list): The constant vector.
        
    Returns:
        list: The solution vector x.
    """
    return back_substitution(*gaussian_elimination(A, b))