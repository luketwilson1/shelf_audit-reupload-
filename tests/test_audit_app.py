import unittest
from shelf_audit.audit_app import safe_result,RESULT_ROOT


class AuditAppTests(unittest.TestCase):
    def test_rejects_path_escape(self):
        for path in ['../README.md','%2e%2e/README.md','..\\README.md']:
            with self.assertRaises(ValueError):safe_result(path)

    def test_result_within_output_root(self):
        self.assertEqual(safe_result('fresh_evaluation/summary.json'),(RESULT_ROOT/'fresh_evaluation/summary.json').resolve())
