import unittest
from linear_solver.core import gaussian_elimination, back_substitution, solve_linear_system

class TestLinearSolver(unittest.TestCase):

    def test_gaussian_elimination(self):
        """Test Gaussian elimination."""
        A = [[2, 1, -1],
             [-3, -1, 2],
             [-2, 1, 2]]
        b = [8, -11, -3]
        
        try:
            A_ref, b_ref = gaussian_elimination(A, b)
        except NotImplementedError:
            self.skipTest("gaussian_elimination not implemented")
            
        # Add assertions here once implemented

    def test_back_substitution(self):
        """Test back substitution."""
        # Upper triangular system
        A = [[2, 1, -1],
             [0, 0.5, 0.5],
             [0, 0, -1]]
        b = [8, 1, 1]
        expected_x = [2, 3, -1]
        
        try:
            x = back_substitution(A, b)
        except NotImplementedError:
            self.skipTest("back_substitution not implemented")
            
        self.assertEqual(x, expected_x)

    def test_solve_linear_system(self):
        """Test full linear system solver."""
        A = [[2, 1, -1],
             [-3, -1, 2],
             [-2, 1, 2]]
        b = [8, -11, -3]
        expected_x = [2, 3, -1]
        
        try:
            x = solve_linear_system(A, b)
        except NotImplementedError:
            self.skipTest("solve_linear_system not implemented")
            
        self.assertEqual(x, expected_x)

if __name__ == '__main__':
    unittest.main()
