import unittest
from vector_algebra.core import vector_add, scalar_multiply, linear_combination

class TestVectorAlgebra(unittest.TestCase):

    def test_vector_add(self):
        """Test vector addition."""
        v1 = [1, 2, 3]
        v2 = [4, 5, 6]
        expected = [5, 7, 9]
        
        try:
            result = vector_add(v1, v2)
        except NotImplementedError:
            self.skipTest("vector_add not implemented")
            
        self.assertEqual(result, expected)
        self.assertEqual(vector_add([0, 0], [0, 0]), [0, 0])
        self.assertEqual(vector_add([-1, 1], [1, -1]), [0, 0])

    def test_scalar_multiply(self):
        """Test scalar multiplication."""
        c = 2
        v = [1, 2, 3]
        expected = [2, 4, 6]
        
        try:
            result = scalar_multiply(c, v)
        except NotImplementedError:
            self.skipTest("scalar_multiply not implemented")
            
        self.assertEqual(result, expected)
        self.assertEqual(scalar_multiply(0, [1, 2, 3]), [0, 0, 0])
        self.assertEqual(scalar_multiply(-1, [1, -2]), [-1, 2])

    def test_linear_combination(self):
        """Test linear combination."""
        c = 2
        v = [1, 0]
        d = 3
        w = [0, 1]
        expected = [2, 3]
        
        try:
            result = linear_combination(c, v, d, w)
        except NotImplementedError:
            self.skipTest("linear_combination not implemented")
            
        self.assertEqual(result, expected)
        self.assertEqual(linear_combination(1, [1, 2], 1, [3, 4]), [4, 6])

if __name__ == '__main__':
    unittest.main()
