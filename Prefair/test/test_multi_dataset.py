import sys
sys.path.append("../src")
# print(sys.path)
import unittest
from mbi import Dataset, Domain
import numpy as np

class TestMultiDatasets(unittest.TestCase):

    def setUp(self):
        attrs = ['a','b','c','d']
        shape = [3,4,5,6]
        domain = Domain(attrs, shape)
        self.data = Dataset.synthetic(domain, 5)

    def test_product(self):
        proj_a = self.data.project(['a'])
        ans_a = Domain(['a'],[3])
        self.assertEqual(proj_a.domain, ans_a)

        proj_b = self.data.project(['b'])
        ans_b = Domain(['b'],[4])
        self.assertEqual(proj_b.domain, ans_b)
        product_ab = proj_a * proj_b
        prod_ans = [[i*j for j in proj_b.datavector()] for i in proj_a.datavector()]

        self.assertTrue(np.array_equal(product_ab, np.array(prod_ans).flatten()))

    def test_datavector(self):
        vec = self.data.datavector()
        self.assertTrue(vec.size, 3*4*5*6)

if __name__ == '__main__':
    unittest.main()