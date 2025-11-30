"""
Core functions for the Vector Algebra Toolkit.
"""

def vector_add(v1, v2):
    """
    Adds two vectors (lists of numbers) element-wise.
    
    Args:
        v1 (list): The first vector.
        v2 (list): The second vector.
        
    Returns:
        list: A new vector representing the sum of v1 and v2.
    """
    return [v1i + v2i for v1i, v2i in zip(v1, v2) ]

def scalar_multiply(c, v):
    """
    Multiplies a vector by a scalar.
    
    Args:
        c (float or int): The scalar value.
        v (list): The vector.
        
    Returns:
        list: A new vector representing the scalar multiplication c * v.
    """
    return [c * vi for vi in v]

def linear_combination(c, v, d, w):
    """
    Calculates the linear combination c*v + d*w.
    
    Args:
        c (float or int): Scalar for v.
        v (list): First vector.
        d (float or int): Scalar for w.
        w (list): Second vector.
        
    Returns:
        list: A new vector representing the linear combination.
    """
    return vector_add(scalar_multiply(c, v), scalar_multiply(d, w))