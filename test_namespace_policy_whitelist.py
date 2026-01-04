import unittest

import namespace_policy


class TestNamespacePolicyDynamicWhitelist(unittest.TestCase):
    def tearDown(self) -> None:
        # Reset both dynamic lists after each test
        namespace_policy.set_dynamic_blacklist(set())
        namespace_policy.set_dynamic_whitelist(set())

    def test_dynamic_whitelist_updates_classification(self):
        # Add to dynamic whitelist
        namespace_policy.set_dynamic_whitelist({"test-trusted-user"})
        decision, reason = namespace_policy.classify_namespace("test-trusted-user")
        self.assertEqual(decision, "allow_whitelist")
        self.assertEqual(reason, "allow:whitelisted_namespace")

    def test_dynamic_whitelist_normalization(self):
        # Test that normalization works
        namespace_policy.set_dynamic_whitelist({"Test-User", "ANOTHER-User"})
        
        # Lowercase should match
        decision, _ = namespace_policy.classify_namespace("test-user")
        self.assertEqual(decision, "allow_whitelist")
        
        decision, _ = namespace_policy.classify_namespace("another-user")
        self.assertEqual(decision, "allow_whitelist")

    def test_blacklist_wins_over_whitelist(self):
        # If a namespace is both blacklisted and whitelisted, blacklist should win
        namespace_policy.set_dynamic_whitelist({"conflicted-user"})
        namespace_policy.set_dynamic_blacklist({"conflicted-user"})
        
        decision, reason = namespace_policy.classify_namespace("conflicted-user")
        self.assertEqual(decision, "deny_blacklist")
        self.assertEqual(reason, "skip:blacklisted_namespace")

    def test_base_whitelist_still_works(self):
        # Base whitelist should still work
        # Pick a namespace that's in BASE_WHITELIST (from config)
        base_wl = namespace_policy.BASE_WHITELIST
        if base_wl:
            test_ns = next(iter(base_wl))
            decision, reason = namespace_policy.classify_namespace(test_ns)
            self.assertEqual(decision, "allow_whitelist")
            self.assertEqual(reason, "allow:whitelisted_namespace")

    def test_get_dynamic_whitelist(self):
        test_set = {"user1", "user2", "user3"}
        namespace_policy.set_dynamic_whitelist(test_set)
        
        retrieved = namespace_policy.get_dynamic_whitelist()
        self.assertEqual(retrieved, test_set)

    def test_get_whitelist_includes_both(self):
        # Add some to dynamic whitelist
        namespace_policy.set_dynamic_whitelist({"dynamic1", "dynamic2"})
        
        # Get combined whitelist
        combined = namespace_policy.get_whitelist()
        
        # Should include dynamic entries
        self.assertIn("dynamic1", combined)
        self.assertIn("dynamic2", combined)
        
        # Should also include base whitelist entries
        base_wl = namespace_policy.BASE_WHITELIST
        for ns in base_wl:
            self.assertIn(ns, combined)

    def test_empty_dynamic_whitelist(self):
        namespace_policy.set_dynamic_whitelist(set())
        
        # Regular namespace should get "allow" (not whitelisted)
        decision, reason = namespace_policy.classify_namespace("random-user")
        self.assertEqual(decision, "allow")
        self.assertIsNone(reason)

    def test_dynamic_whitelist_with_url(self):
        # Test that URL-based classification works with dynamic whitelist
        namespace_policy.set_dynamic_whitelist({"test-user"})
        
        decision, _ = namespace_policy.classify_namespace("https://huggingface.co/test-user/model-name")
        self.assertEqual(decision, "allow_whitelist")

    def test_dynamic_whitelist_with_repo_id(self):
        # Test that repo-id based classification works with dynamic whitelist
        namespace_policy.set_dynamic_whitelist({"test-user"})
        
        decision, _ = namespace_policy.classify_namespace("test-user/model-name")
        self.assertEqual(decision, "allow_whitelist")


if __name__ == "__main__":
    unittest.main()
