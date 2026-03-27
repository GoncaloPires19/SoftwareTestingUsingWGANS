#!/usr/bin/env python
"""
Validation Script for Repository Refactoring
Verifies that all essential components are accessible and functional after optimization.

Usage: python validate_refactoring.py
"""

import sys
import os
from pathlib import Path
import subprocess

def check_file_exists(filepath: str, description: str) -> bool:
    """Check if a file exists and report."""
    path = Path(filepath)
    exists = path.exists()
    status = "✅" if exists else "❌"
    print(f"{status} {description}: {filepath}")
    return exists

def check_module_importable(module_name: str, description: str) -> bool:
    """Check if a Python module can be imported."""
    try:
        __import__(module_name)
        print(f"✅ {description}: {module_name}")
        return True
    except ImportError as e:
        print(f"❌ {description}: {module_name} - {e}")
        return False

def check_dependency_installed(package_name: str) -> bool:
    """Check if a pip package is installed."""
    try:
        __import__(package_name)
        return True
    except ImportError:
        return False

def main():
    """Run all validation checks."""
    print("=" * 70)
    print("REPOSITORY REFACTORING VALIDATION")
    print("=" * 70)
    
    checks_passed = 0
    checks_total = 0
    
    # 1. Check Essential Files
    print("\n📁 ESSENTIAL FILES CHECK")
    print("-" * 70)
    
    essential_files = [
        ("requirements.txt", "Python Dependencies"),
        ("pom.xml", "Maven Build Config"),
        (".gitignore", "Git Ignore Config"),
        ("python/scripts/tools/cleanup.py", "Cleanup Tool"),
        ("python/scripts/tools/monitor_training.py", "Monitor Tool"),
        ("python/src/model.py", "WGAN Model"),
        ("python/src/config.py", "Configuration"),
    ]
    
    for filepath, desc in essential_files:
        checks_total += 1
        if check_file_exists(filepath, desc):
            checks_passed += 1
    
    # 2. Check Python Imports
    print("\n🐍 PYTHON MODULES CHECK")
    print("-" * 70)
    
    modules = [
        ("torch", "PyTorch"),
        ("pandas", "Pandas"),
        ("numpy", "NumPy"),
        ("sklearn", "Scikit-Learn"),
        ("scipy", "SciPy (Recently Added)"),
        ("matplotlib", "Matplotlib"),
        ("seaborn", "Seaborn"),
    ]
    
    for module, desc in modules:
        checks_total += 1
        if check_module_importable(module, desc):
            checks_passed += 1
        else:
            print(f"  💡 Install with: pip install -r requirements.txt")
    
    # 3. Check Data Directories
    print("\n📊 DATA STRUCTURES CHECK")
    print("-" * 70)
    
    data_dirs = [
        ("data/raw", "Raw Data Directory"),
        ("data/processed", "Processed Data Directory"),
        ("python/data/raw", "Python Raw Data"),
        ("python/data/processed", "Python Processed Data"),
        ("outputs", "Outputs Directory"),
        ("python/scripts/tools", "Tools Directory (NEW)"),
    ]
    
    for dirpath, desc in data_dirs:
        checks_total += 1
        path = Path(dirpath)
        exists = path.exists()
        status = "✅" if exists else "❌"
        print(f"{status} {desc}: {dirpath}")
        if exists:
            checks_passed += 1
    
    # 4. Check .gitignore Rules
    print("\n🔒 GITIGNORE SECURITY CHECK")
    print("-" * 70)
    
    try:
        with open(".gitignore", "r") as f:
            gitignore_content = f.read()
        
        required_patterns = [
            ("*.err", "Error files"),
            ("*.out", "Output text files"),
            ("*.log", "Log files"),
            ("*.pth", "PyTorch models"),
            ("target/", "Maven build artifacts"),
        ]
        
        for pattern, desc in required_patterns:
            checks_total += 1
            if pattern in gitignore_content:
                print(f"✅ {desc}: {pattern}")
                checks_passed += 1
            else:
                print(f"❌ {desc}: {pattern} NOT FOUND")
    except Exception as e:
        print(f"❌ Error reading .gitignore: {e}")
    
    # 5. Check requirements.txt
    print("\n📦 DEPENDENCIES CHECK")
    print("-" * 70)
    
    try:
        with open("requirements.txt", "r") as f:
            requirements = f.read()
        
        critical_deps = [
            ("torch==2.0.1", "PyTorch (Fixed Version)"),
            ("scipy>=1.9.0", "SciPy (Recently Added)"),
            ("pandas>=1.5.0", "Pandas"),
        ]
        
        for dep, desc in critical_deps:
            checks_total += 1
            if dep in requirements:
                print(f"✅ {desc}: {dep}")
                checks_passed += 1
            else:
                print(f"❌ {desc}: {dep} NOT FOUND")
    except Exception as e:
        print(f"❌ Error reading requirements.txt: {e}")
    
    # 6. Check Git Status
    print("\n🔄 GIT STATUS CHECK")
    print("-" * 70)
    
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=".",
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            modified_files = result.stdout.strip().split("\n") if result.stdout.strip() else []
            checks_total += 1
            print(f"✅ Git repository accessible")
            checks_passed += 1
            
            if modified_files and modified_files[0]:
                print(f"  📝 Modified files detected: {len(modified_files)}")
                for f in modified_files[:5]:  # Show first 5
                    print(f"     {f}")
                if len(modified_files) > 5:
                    print(f"     ... and {len(modified_files)-5} more")
        else:
            print(f"❌ Git repository error")
    except Exception as e:
        print(f"⚠️  Could not check git status: {e}")
    
    # Summary
    print("\n" + "=" * 70)
    print(f"VALIDATION SUMMARY: {checks_passed}/{checks_total} Checks Passed")
    print("=" * 70)
    
    if checks_passed == checks_total:
        print("✅ ALL SYSTEMS GO! Repository is properly optimized.")
        return 0
    else:
        percentage = (checks_passed / checks_total) * 100
        print(f"⚠️  {percentage:.1f}% checks passed. Review failures above.")
        print("\n💡 Next Steps:")
        print("   1. Install dependencies: pip install -r requirements.txt")
        print("   2. Verify imports work correctly")
        print("   3. Run: python python/scripts/validate_before_training.py")
        return 1

if __name__ == "__main__":
    sys.exit(main())
