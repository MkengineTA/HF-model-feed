"""
Manual integration test for dynamic whitelist feature.
Tests the full flow without needing HuggingFace API access.
"""

import tempfile
import os
from datetime import datetime, timezone

from database import Database
import namespace_policy
from run_stats import RunStats
from reporter import Reporter
import config

def test_dynamic_whitelist_flow():
    print("=" * 60)
    print("Manual Integration Test: Dynamic Whitelist Feature")
    print("=" * 60)
    
    # Create temporary database
    temp_fd, temp_db_path = tempfile.mkstemp(suffix=".db")
    temp_dir = tempfile.mkdtemp()
    
    try:
        print("\n1. Testing Database Operations")
        print("-" * 40)
        
        db = Database(temp_db_path)
        
        # Add some namespaces to dynamic whitelist
        print("Adding namespaces to dynamic whitelist...")
        db.upsert_dynamic_whitelist({
            "test-org-1": 1,
            "test-org-2": 2,
            "test-org-3": 1,
        }, reason="tier3_org")
        
        wl = db.get_dynamic_whitelist()
        print(f"✓ Dynamic whitelist contains: {wl}")
        assert wl == {"test-org-1", "test-org-2", "test-org-3"}, "Whitelist mismatch"
        
        # Test removal
        print("Removing one namespace...")
        db.remove_dynamic_whitelist({"test-org-2"})
        wl = db.get_dynamic_whitelist()
        print(f"✓ After removal: {wl}")
        assert wl == {"test-org-1", "test-org-3"}, "Removal failed"
        
        print("\n2. Testing Namespace Policy Integration")
        print("-" * 40)
        
        # Load dynamic whitelist into namespace policy
        namespace_policy.set_dynamic_whitelist(wl)
        
        # Test whitelisted namespace
        decision, reason = namespace_policy.classify_namespace("test-org-1")
        print(f"✓ test-org-1 classification: {decision} ({reason})")
        assert decision == "allow_whitelist", "Should be whitelisted"
        
        # Test non-whitelisted namespace
        decision, reason = namespace_policy.classify_namespace("random-user")
        print(f"✓ random-user classification: {decision}")
        assert decision == "allow", "Should be allowed but not whitelisted"
        
        # Test blacklist wins
        namespace_policy.set_dynamic_blacklist({"test-org-1"})
        decision, reason = namespace_policy.classify_namespace("test-org-1")
        print(f"✓ test-org-1 (blacklisted) classification: {decision} ({reason})")
        assert decision == "deny_blacklist", "Blacklist should win"
        
        # Reset blacklist
        namespace_policy.set_dynamic_blacklist(set())
        
        print("\n3. Testing Run Stats Tier 2 Tracking")
        print("-" * 40)
        
        stats = RunStats()
        
        # Add some Tier 2 candidates
        print("Recording Tier 2 candidates...")
        stats.record_tier2_candidate(
            namespace="user1",
            followers=250,
            is_pro=False,
            model_id="user1/model-1"
        )
        stats.record_tier2_candidate(
            namespace="user1",
            followers=250,
            is_pro=False,
            model_id="user1/model-2"
        )
        stats.record_tier2_candidate(
            namespace="user2",
            followers=500,
            is_pro=True,
            model_id="user2/model-1"
        )
        
        print(f"✓ Tier 2 candidates: {list(stats.tier2_candidates.keys())}")
        assert "user1" in stats.tier2_candidates, "user1 should be tracked"
        assert "user2" in stats.tier2_candidates, "user2 should be tracked"
        assert stats.tier2_candidates["user1"]["count"] == 2, "user1 count should be 2"
        assert stats.tier2_candidates["user2"]["is_pro"] == True, "user2 should be PRO"
        
        print("\n4. Testing Reporter Tier 2 Section")
        print("-" * 40)
        
        # Create a test report
        reporter = Reporter(output_dir=temp_dir)
        
        # Temporarily enable the feature
        original_value = config.REPORT_INCLUDE_TIER2_REVIEW
        config.REPORT_INCLUDE_TIER2_REVIEW = True
        
        try:
            print("Generating report with Tier 2 section...")
            path = reporter.write_markdown_report(
                stats=stats,
                processed_models=[],
                date_str="2024-01-01"
            )
            
            content = path.read_text(encoding="utf-8")
            
            # Verify Tier 2 section exists
            assert "## Tier 2 whitelist candidates (review)" in content, "Tier 2 section missing"
            assert "user1" in content, "user1 should be in report"
            assert "user2" in content, "user2 should be in report"
            assert "250 followers" in content, "Follower count missing"
            assert "PRO" in content, "PRO badge missing"
            assert "2 model(s) this run" in content, "Model count missing"
            
            print(f"✓ Report generated at: {path}")
            print("✓ Tier 2 section contains expected content")
            
            # Print a snippet of the Tier 2 section
            start = content.find("## Tier 2 whitelist candidates")
            end = content.find("## Processed models", start)
            tier2_section = content[start:end] if start != -1 and end != -1 else ""
            if tier2_section:
                print("\nTier 2 Section Preview:")
                print("-" * 40)
                print(tier2_section[:500])
                
        finally:
            config.REPORT_INCLUDE_TIER2_REVIEW = original_value
        
        print("\n" + "=" * 60)
        print("✓ ALL TESTS PASSED")
        print("=" * 60)
        
        db.close()
        
    finally:
        # Cleanup
        os.close(temp_fd)
        os.unlink(temp_db_path)
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)

if __name__ == "__main__":
    test_dynamic_whitelist_flow()
