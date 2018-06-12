import unittest

from main import run
import os


class MainTest(unittest.TestCase):
    def setUp(self):
        self.db_file = 'tmp_unit_tests.db'
        self.args = ['--num-gen', '10',
                     '--pop-size', '10',
                     '--db-file', self.db_file]

    def test_ga(self):
        self.args += ['test_scenario/dataset.csv', 'TEST']
        run(args=self.args)

    def test_pso(self):
        self.args += ['test_scenario/dataset.csv', 'TEST',
                      '--strategy', 'pso']
        run(args=self.args)

    def tearDown(self):
        os.remove(self.db_file)


if __name__ == '__main__':
    unittest.main()
