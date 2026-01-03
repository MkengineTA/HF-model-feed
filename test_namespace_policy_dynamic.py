import unittest

import namespace_policy


class TestNamespacePolicyDynamic(unittest.TestCase):
    def tearDown(self) -> None:
        namespace_policy.set_dynamic_blacklist(set())

    def test_dynamic_blacklist_updates_classification(self):
        namespace_policy.set_dynamic_blacklist({"tempUploader"})
        decision, reason = namespace_policy.classify_namespace("tempUploader")
        self.assertEqual(decision, "deny_blacklist")
        self.assertEqual(reason, "skip:blacklisted_namespace")


if __name__ == "__main__":
    unittest.main()
