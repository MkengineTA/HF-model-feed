import unittest

from main import tree_has_readme


class TestTreeHasReadme(unittest.TestCase):
    def test_none_or_non_list(self):
        self.assertFalse(tree_has_readme(None))
        self.assertFalse(tree_has_readme("not-a-list"))

    def test_root_readme(self):
        files = [{"path": "README.md"}]
        self.assertTrue(tree_has_readme(files))

    def test_nested_readme_variants(self):
        files = [
            {"path": "docs/readme.txt"},
            {"path": "nested/Readme.md"},
            {"path": "cards/MODELCARD.md"},
        ]
        self.assertTrue(tree_has_readme(files))

    def test_no_readme(self):
        files = [{"path": "weights.bin"}, {"path": "config.json"}]
        self.assertFalse(tree_has_readme(files))

    def test_empty_paths(self):
        files = [{"path": ""}, {"path": None}]
        self.assertFalse(tree_has_readme(files))


if __name__ == "__main__":
    unittest.main()
