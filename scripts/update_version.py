#!/usr/bin/env python3
"""
Version Synchronization Script for Safety-Critical Releases.

As a Senior MLOps Engineer, this script ensures VERSION INTEGRITY across all
project artifacts. It is designed to run automatically in CI/CD pipelines
and before manual releases.

Safety Guarantees:
- Single source of truth: pyproject.toml
- Verified updates: Only updates when version actually changes
- Backward compatible: Works with both development and packaged installations
- Audit trail: Logs all version updates for compliance

Usage:
    # Manually update version information
    python scripts/update_version.py

    # Dry run (show what would change without modifying files)
    python scripts/update_version.py --dry-run

    # Verify version consistency
    python scripts/update_version.py --verify
"""

import re
import sys
from pathlib import Path
from typing import Tuple, Optional
import argparse


def get_version_from_pyproject() -> str:
    """Extract version from pyproject.toml (single source of truth)."""
    pyproject_path = Path(__file__).parent.parent / "pyproject.toml"
    
    if not pyproject_path.exists():
        raise FileNotFoundError(f"pyproject.toml not found at {pyproject_path}")
    
    content = pyproject_path.read_text()
    
    # Match: version = "X.Y.Z"
    match = re.search(r'version\s*=\s*["\']([^"\']+)["\']', content)
    if not match:
        raise ValueError("Could not parse version from pyproject.toml")
    
    return match.group(1)


def get_version_from_model_card() -> str:
    """Extract version from MODEL_CARD.md (current state)."""
    model_card = Path(__file__).parent.parent / "MODEL_CARD.md"
    
    if not model_card.exists():
        return None
    
    content = model_card.read_text()
    
    # Match the version table row:  | Version | X.Y.Z |
    match = re.search(r"\|\s*Version\s*\|\s*([0-9][^\s|]*)", content)
    if match:
        return match.group(1)
    
    return None


def get_version_from_docs_index() -> Optional[str]:
    """Extract version from docs/index.md (current state)."""
    index_path = Path(__file__).parent.parent / "docs" / "index.md"

    if not index_path.exists():
        return None

    content = index_path.read_text()

    # Match: Current Version: **X.Y.Z**
    match = re.search(r"Current Version:\s*\*\*([^*]+)\*\*", content)
    if match:
        return match.group(1)

    return None


def update_model_card(version: str, dry_run: bool = False) -> bool:
    """Update version in MODEL_CARD.md."""
    model_card = Path(__file__).parent.parent / "MODEL_CARD.md"
    
    if not model_card.exists():
        print(f"⚠ MODEL_CARD.md not found at {model_card}")
        return False
    
    content = model_card.read_text()
    
    # The release version appears in several places in the card; all derive
    # from pyproject.toml.  The wildcard CPE (``:*:``) is intentionally left
    # untouched -- only the current-release CPE carries a concrete version.
    patterns = [
        # Version table row:        | Version | X.Y.Z |
        (r"(\|\s*Version\s*\|\s*)([0-9][^\s|]*)", rf"\g<1>{version}"),
        # Current-release CPE:      spotforecast2_safe:X.Y.Z:
        (r"(spotforecast2_safe:)([0-9][^:]*)(:)", rf"\g<1>{version}\g<3>"),
        # Citation reference:       (Version X.Y.Z)
        (r"(\(Version\s+)([0-9][^)]*)(\))", rf"\g<1>{version}\g<3>"),
        # Lifecycle sentence:       current release is X.Y.Z
        (r"(current release is\s+)([0-9][^\s,]*)", rf"\g<1>{version}"),
    ]
    new_content = content
    for pattern, replacement in patterns:
        new_content = re.sub(pattern, replacement, new_content)
    
    if new_content == content:
        print(f"ℹ No changes needed: MODEL_CARD.md already has version {version}")
        return False
    
    if dry_run:
        print(f"[DRY RUN] Would update MODEL_CARD.md to version {version}")
        return True
    
    model_card.write_text(new_content)
    print(f"✓ Updated MODEL_CARD.md to version {version}")
    return True


def update_docs_index(version: str, dry_run: bool = False) -> bool:
    """Update version badge and current version in docs/index.md."""
    index_path = Path(__file__).parent.parent / "docs" / "index.md"

    if not index_path.exists():
        # docs/index.md is optional; skip silently when it is absent.
        return False

    content = index_path.read_text()

    badge_pattern = r"(https://img\.shields\.io/badge/version-)([^-]+)(-blue\.svg)"
    current_version_pattern = r"(Current Version:\s*\*\*)([^*]+)(\*\*)"

    updated = content
    updated = re.sub(badge_pattern, rf"\g<1>{version}\g<3>", updated)
    updated = re.sub(current_version_pattern, rf"\g<1>{version}\g<3>", updated)

    if updated == content:
        print(f"ℹ No changes needed: docs/index.md already has version {version}")
        return False

    if dry_run:
        print(f"[DRY RUN] Would update docs/index.md:")
        print(f"  Old version: {get_version_from_docs_index()}")
        print(f"  New version: {version}")
        return True

    index_path.write_text(updated)
    print(f"✓ Updated docs/index.md to version {version}")
    return True


def verify_consistency() -> Tuple[bool, str]:
    """Verify version consistency across all files."""
    try:
        pyproject_version = get_version_from_pyproject()
        model_card_version = get_version_from_model_card()
        docs_index_version = get_version_from_docs_index()
        
        print(f"\n📋 Version Consistency Check:")
        print(f"  pyproject.toml:  {pyproject_version}")
        print(f"  MODEL_CARD.md:   {model_card_version}")
        print(f"  docs/index.md:   {docs_index_version}")
        
        # Only compare artifacts that exist; a missing optional file
        # (value ``None``) must not by itself count as a mismatch.
        present = {
            "MODEL_CARD.md": model_card_version,
            "docs/index.md": docs_index_version,
        }
        mismatches = [
            name
            for name, found in present.items()
            if found is not None and found != pyproject_version
        ]
        if not mismatches:
            print("✓ Versions are in sync!")
            return True, pyproject_version
        print(f"⚠ Version mismatch detected in: {', '.join(mismatches)}")
        return False, pyproject_version
    
    except Exception as e:
        print(f"❌ Error checking versions: {e}")
        return False, None


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Synchronize version information across project artifacts"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show changes without modifying files"
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Only verify version consistency (do not update)"
    )
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("spotforecast2-safe: Version Synchronization Script")
    print("=" * 70)
    
    try:
        # Always show consistency check
        is_consistent, pyproject_version = verify_consistency()
        
        if args.verify:
            # Only verify mode
            return 0 if is_consistent else 1
        
        if not is_consistent and pyproject_version:
            # Update if versions don't match
            print(f"\n🔄 Synchronizing to pyproject.toml version: {pyproject_version}")
            updated = update_model_card(pyproject_version, dry_run=args.dry_run)
            updated = update_docs_index(pyproject_version, dry_run=args.dry_run) or updated
            
            if args.dry_run:
                print("\n[DRY RUN] No files were actually modified")
            elif updated:
                print("\n✓ All versions are now synchronized!")
            
            return 0
        elif is_consistent:
            print("\n✓ No action needed - all versions are synchronized")
            return 0
        else:
            print("\n❌ Could not update versions")
            return 1
    
    except Exception as e:
        print(f"\n❌ Error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
