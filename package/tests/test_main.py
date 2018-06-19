import unittest
from typing import List

from package.main import run
import os
import glob

from package.orm_models import local_create_session, Result


class MainTest(unittest.TestCase):
    def setUp(self):
        self.exp_name = 'TEST'
        self.db_file = 'tmp_unit_tests.db'
        self.args = ['--num-gen', '10',
                     '--pop-size', '10',
                     '--db-file', self.db_file]
        print(glob.glob('./*'))
        self.testing_dataset = './test_dataset/dataset.csv'

    def test_ga(self):
        self.args += [
            self.testing_dataset, self.exp_name
        ]
        run(args=self.args)

    def test_pso(self):
        self.args += [
            self.testing_dataset, self.exp_name,
            '--strategy', 'pso'
        ]
        run(args=self.args)

    def test_feature_limit_ga(self):
        self.args += [
            self.testing_dataset, self.exp_name,
            '--max-features', '50'
        ]
        result_ids = run(args=self.args)
        result_id = result_ids[0]

        session = local_create_session(self.db_file)
        result: Result = session.query(Result).filter(Result.id== result_id).first()
        features = result.selected_features
        session.close()

        self.assertLessEqual(len(features), 50)

    def tearDown(self):
        os.remove(self.db_file)


if __name__ == '__main__':
    unittest.main()
