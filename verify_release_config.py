#!/usr/bin/env python3
"""
Verification script for spotforecast2-safe release configuration.

As a Senior MLOps Engineer, this ensures all safety-critical configurations
are properly set up before production releases.
"""

import json
import sys
from pathlib import Path


def check_pyproject_toml():
    """Verify pyproject.toml has correct package name."""
    pyproject = Path("pyproject.toml")
    if not pyproject.exists():
        print("❌ pyproject.toml not found")
        return False
    
    content = pyproject.read_text()
    if 'name = "spotforecast2-safe"' in content:
        print("✓ Package name correct in pyproject.toml: spotforecast2-safe")
        return True
    else:
        print("❌ Package name incorrect in pyproject.toml")
        print("   Expected: name = \"spotforecast2-safe\"")
        return False


def check_releaserc_json():
    """Verify .releaserc.json has correct PyPI configuration."""
    releaserc = Path(".releaserc.json")
    if not releaserc.exists():
        print("❌ .releaserc.json not found")
        return False
    
    try:
        config = json.loads(releaserc.read_text())
        
        # Check branches
        branches = config.get("branches", [])
        if not isinstance(branches, list) or len(branches) == 0:
            print("❌ Invalid branches configuration in .releaserc.json")
            return False
        print(f"✓ Release branches configured: {[b.get('name', b) for b in branches]}")
        
        # Check plugins
        plugins = config.get("plugins", [])
        has_pypi_publish = False
        
        for plugin in plugins:
            if isinstance(plugin, list) and len(plugin) > 1:
                if plugin[0] == "@semantic-release/exec":
                    config_section = plugin[1] if len(plugin) > 1 else {}
                    publish_cmd = config_section.get("publishCmd", "")
                    if "twine upload" in publish_cmd and "PYPI_TOKEN" in publish_cmd:
                        has_pypi_publish = True
                        print(f"✓ PyPI publishing configured: twine with PYPI_TOKEN")
                        break
        
        if not has_pypi_publish:
            print("⚠ PyPI publishing might not be properly configured")
            return False
        
        return True
        
    except json.JSONDecodeError as e:
        print(f"❌ Invalid JSON in .releaserc.json: {e}")
        return False


def check_workflows():
    """Verify GitHub workflows are properly configured."""
    workflows_dir = Path(".github/workflows")
    
    if not workflows_dir.exists():
        print("❌ .github/workflows directory not found")
        return False
    
    required_workflows = {
        "ci.yml": ["pytest", "spotforecast2_safe"],
        "release.yml": ["semantic-release", "PYPI_TOKEN"],
        "docs.yml": ["mkdocs"],
    }
    
    all_ok = True
    for workflow_name, required_patterns in required_workflows.items():
        workflow_path = workflows_dir / workflow_name
        if not workflow_path.exists():
            print(f"❌ {workflow_name} not found")
            all_ok = False
            continue
        
        content = workflow_path.read_text()
        missing = [pat for pat in required_patterns if pat not in content]
        
        if missing:
            print(f"❌ {workflow_name} missing: {missing}")
            all_ok = False
        else:
            print(f"✓ {workflow_name} properly configured")
    
    return all_ok


def check_src_structure():
    """Verify source code structure uses spotforecast2_safe."""
    src_dir = Path("src/spotforecast2_safe")
    
    if not src_dir.exists():
        print("❌ src/spotforecast2_safe directory not found")
        return False
    
    if not (src_dir / "__init__.py").exists():
        print("❌ src/spotforecast2_safe/__init__.py not found")
        return False
    
    print("✓ Source code structure correct: src/spotforecast2_safe")
    return True


def check_test_imports():
    """Verify test files use spotforecast2_safe imports."""
    tests_dir = Path("tests")
    
    if not tests_dir.exists():
        print("❌ tests directory not found")
        return False
    
    # Sample check - verify at least some tests import spotforecast2_safe
    test_files = list(tests_dir.glob("test_*.py"))
    if not test_files:
        print("⚠ No test files found")
        return False
    
    # Check first few test files
    files_checked = 0
    files_with_safe_imports = 0
    
    for test_file in test_files[:5]:
        content = test_file.read_text()
        files_checked += 1
        
        if "spotforecast2_safe" in content:
            files_with_safe_imports += 1
        elif "spotforecast2" in content and "spotforecast2_safe" not in content:
            print(f"❌ {test_file.name} has old package references")
            return False
    
    if files_with_safe_imports >= files_checked:
        print(f"✓ Test imports verified: spotforecast2_safe")
        return True
    else:
        print("⚠ Some test files may not be updated")
        return False


def main():
    """Run all verification checks."""
    print("=" * 70)
    print("spotforecast2-safe: Release Configuration Verification")
    print("=" * 70)
    print()
    
    checks = [
        ("Package Name (pyproject.toml)", check_pyproject_toml),
        ("Release Configuration (.releaserc.json)", check_releaserc_json),
        ("GitHub Workflows", check_workflows),
        ("Source Code Structure", check_src_structure),
        ("Test Imports", check_test_imports),
    ]
    
    results = []
    for check_name, check_func in checks:
        print(f"\n📋 Checking: {check_name}")
        print("-" * 70)
        try:
            result = check_func()
            results.append((check_name, result))
        except Exception as e:
            print(f"❌ Error during check: {e}")
            results.append((check_name, False))
    
    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for check_name, result in results:
        status = "✓ PASS" if result else "❌ FAIL"
        print(f"{status}: {check_name}")
    
    print()
    print(f"Result: {passed}/{total} checks passed")
    print()
    
    if passed == total:
        print("🎉 All configurations verified! Ready for release.")
        print()
        print("Next steps:")
        print("1. Ensure PYPI_TOKEN is added to GitHub secrets")
        print("2. Run a test release or monitor next push to main branch")
        print("3. Verify package appears on PyPI: https://pypi.org/project/spotforecast2-safe/")
        return 0
    else:
        print("⚠️  Some checks failed. Please review the errors above.")
        print()
        print("See PYPI_SETUP.md for detailed setup instructions.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
