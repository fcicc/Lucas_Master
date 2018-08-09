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
        self.args = ['--num-gen', '2',
                     '--pop-size', '2',
                     '--db-file', self.db_file]
        self.testing_datasets = glob.glob('./datasets/*/dataset.csv')

    def test_ga(self):
        for dataset in self.testing_datasets:
            args = self.args + [
                dataset, self.exp_name
            ]
            run(args=args)

    def test_pso(self):
        for dataset in self.testing_datasets:
            args = self.args + [
                dataset, self.exp_name,
                '--strategy', 'pso'
            ]
            run(args=args)

    def test_multiple_run(self):
        exp_name = 'TEST_MULTIPLE'
        n_runs = 2
        self.args += [
            '--run-multiple', str(n_runs),
            self.testing_datasets[0], exp_name
        ]
        run(args=self.args)

        session = local_create_session(self.db_file)
        results = session.query(Result).filter(Result.name == exp_name).all()
        n_results = len(results)
        session.close()

        self.assertEqual(n_runs, n_results)

    def test_feature_limit_ga(self):
        for dataset in self.testing_datasets:
            args = self.args + [
                dataset, self.exp_name,
                '--max-features', '50'
            ]
            result_ids = run(args=args)
            result_id = result_ids[0]

            session = local_create_session(self.db_file)
            result: Result = session.query(Result).filter(Result.id == result_id).first()
            features = result.selected_features
            session.close()

            self.assertLessEqual(len(features), 50)

    def tearDown(self):
        os.remove(self.db_file)


if __name__ == '__main__':
    unittest.main()
