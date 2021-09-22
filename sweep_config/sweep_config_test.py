import sys
sys.path.append("sweep_config")

import unittest

from sweep_config_random import sweep_config_random
from sweep_config_grid import sweep_config_grid
from sweep_config_bayes import sweep_config_bayes

class sweepConfigRandomTests(unittest.TestCase):
    
    def test_config_type(self):
        self.assertTrue(isinstance(sweep_config_random,dict))
    
    def test_necessary_keys(self):
        self.assertTrue("method" in sweep_config_random.keys())
        self.assertTrue("parameters" in sweep_config_random.keys())
    
    def test_type_of_necessary_keys(self):
        self.assertTrue(isinstance(sweep_config_random["method"],str))
        self.assertTrue(isinstance(sweep_config_random["parameters"],dict))
    
    def test_necessary_params_value(self):
        # 팀원들과 협의 하에 변경 가능
        params = sweep_config_random["parameters"]
        self.assertTrue("EPOCHS" in params)
        self.assertTrue("BATCH_SIZE" in params)
        self.assertTrue("LEARNING_RATE" in params)
        

class sweepConfigGridTests(unittest.TestCase):
    def test_config_type(self):
        self.assertTrue(isinstance(sweep_config_grid,dict))
    
    def test_necessary_keys(self):
        self.assertTrue("method" in sweep_config_grid.keys())
        self.assertTrue("parameters" in sweep_config_grid.keys())
    
    def test_type_of_necessary_keys(self):
        self.assertTrue(isinstance(sweep_config_grid["method"],str))
        self.assertTrue(isinstance(sweep_config_grid["parameters"],dict))
    
    def test_necessary_params_value(self):
        # 팀원들과 협의 하에 변경 가능
        params = sweep_config_grid["parameters"]
        self.assertTrue("EPOCHS" in params)
        self.assertTrue("BATCH_SIZE" in params)
        self.assertTrue("LEARNING_RATE" in params)

        for key, val in params.items():
            self.assertTrue(isinstance(key,str))
            self.assertTrue(isinstance(val,dict))
            for p_key, p_val in val.items():
                self.assertTrue(p_key in ("value", "values"))
                if p_key == "values":
                    self.assertTrue(isinstance(p_val,(list, tuple)))


class sweepConfigBayesTest(unittest.TestCase):
    def test_config_type(self):
        self.assertTrue(isinstance(sweep_config_bayes,dict))
    
    def test_necessary_keys(self):
        self.assertTrue("method" in sweep_config_bayes.keys())
        self.assertTrue("metric" in sweep_config_bayes.keys())
        self.assertTrue("parameters" in sweep_config_bayes.keys())
    
    def test_type_of_necessary_keys(self):
        self.assertTrue(isinstance(sweep_config_bayes["method"],str))
        self.assertTrue(isinstance(sweep_config_bayes["metric"],dict))
        self.assertTrue(isinstance(sweep_config_bayes["parameters"],dict))
    
    def test_necessary_params_value(self):
        # 팀원들과 협의 하에 변경 가능
        params = sweep_config_bayes["parameters"]
        self.assertTrue("EPOCHS" in params)
        self.assertTrue("BATCH_SIZE" in params)
        self.assertTrue("LEARNING_RATE" in params)
        
        metric = sweep_config_bayes["metric"]
        self.assertTrue("name" in metric)
        self.assertTrue("goal" in metric)
        
        goal = metric["goal"]
        self.assertTrue(goal in ["maximize", "minimize"])
        
        


if __name__ == "__main__":
    unittest.main()
