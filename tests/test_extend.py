import unittest
from src.process import extend_data
import pandas as pd

class TestExtendData(unittest.TestCase):

    def setUp(self):
        self.df = pd.DataFrame({
            'Time': [0, 1, 2, 3, 4, 5],
            'Voltage': [1.5, 1.6, 1.7, 1.8, 1.9, 2.0],
            'is_outlier': [False, False, False, False, False, False]
        })

    def test_extend_data(self):
        final_df, ext_df = extend_data(self.df, target_hour=10)
        
        self.assertIsInstance(final_df, pd.DataFrame)
        self.assertIsInstance(ext_df, pd.DataFrame)
        self.assertGreater(len(ext_df), 0)
        self.assertTrue(all(ext_df['Time'] > self.df['Time'].max()))

    def test_extend_data_with_insufficient_data(self):
        small_df = pd.DataFrame({
            'Time': [0],
            'Voltage': [1.5],
            'is_outlier': [False]
        })
        final_df, ext_df = extend_data(small_df, target_hour=10)
        
        self.assertIsInstance(final_df, pd.DataFrame)
        self.assertEqual(len(ext_df), 0)

if __name__ == '__main__':
    unittest.main()