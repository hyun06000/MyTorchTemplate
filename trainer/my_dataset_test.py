import sys
sys.path.extend(["sweep_config","trainer"])

import unittest

from torchvision import transforms

from my_dataset import myDataset
from sweep_config.config_generator import common_params

# Build myDataset
tr_MD = myDataset(
        name=common_params["DATASET"]["value"],
        data_path=common_params["DATA_PATH"]["value"], 
        train=True, 
        transforms=transforms.ToTensor(), # hyperparameter로 지정된 transform사용
        download=True
    )
te_MD = myDataset(
        name=common_params["DATASET"]["value"],
        data_path=common_params["DATA_PATH"]["value"], 
        train=False, 
        transforms=transforms.ToTensor(), # hyperparameter로 지정된 transform사용
        download=True
    )

class myDatasetTests(unittest.TestCase):
    def test_len_dataset(self):
        self.assertTrue(len(tr_MD) > 0)
        self.assertTrue(len(te_MD) > 0)
    
    def test_data(self):
        for MD in [tr_MD, te_MD]:
            X, y = MD[0]
            img_shape_ref = X.shape
            img_type_ref = type(X.max().item())
            label_type_ref = type(y)

            # channel first test (gray or color)
            self.assertIn(img_shape_ref[0], [1, 3])
            for idx in range(1, len(MD)):
                X, y = MD[idx]
                # image size test
                self.assertEqual(X.shape, img_shape_ref)
                # image type test
                self.assertEqual(type(X.max().item()), img_type_ref)
                # label type test
                self.assertEqual(type(y), label_type_ref)


if __name__ == "__main__":
    unittest.main()
    print("my_dataset_test.py done")