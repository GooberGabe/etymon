#!/usr/bin/env python3
"""
Setup script for Etymon development environment.

This script helps set up the development environment and install dependencies.
"""

import os
import sys
import subprocess


def run_command(command, description):
    """Run a command and handle errors.
    
    Args:
        command: Command to run (as list)
        description: Description of what the command does
    """
    print(f"\n{description}...")
    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        print(f"✓ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ {description} failed:")
        print(f"  Exit code: {e.returncode}")
        print(f"  stdout: {e.stdout}")
        print(f"  stderr: {e.stderr}")
        return False
    except FileNotFoundError:
        print(f"✗ {description} failed: Command not found")
        return False


def check_python_version():
    """Check if Python version is adequate."""
    print("Checking Python version...")
    version = sys.version_info
    
    if version.major >= 3 and version.minor >= 8:
        print(f"✓ Python {version.major}.{version.minor}.{version.micro} is supported")
        return True
    else:
        print(f"✗ Python {version.major}.{version.minor}.{version.micro} is not supported")
        print("  Etymon requires Python 3.8 or later")
        return False


def install_dependencies():
    """Install Python dependencies."""
    if not os.path.exists('requirements.txt'):
        print("✗ requirements.txt not found")
        return False
    
    return run_command(
        [sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'],
        "Installing Python dependencies"
    )


def create_directories():
    """Create necessary directories."""
    dirs = [
        'logs',
        'screenshots',
        'exports'
    ]
    
    print("\nCreating directories...")
    for dir_path in dirs:
        if not os.path.exists(dir_path):
            try:
                os.makedirs(dir_path)
                print(f"  ✓ Created {dir_path}/")
            except OSError as e:
                print(f"  ✗ Failed to create {dir_path}/: {e}")
                return False
        else:
            print(f"  • {dir_path}/ already exists")
    
    return True


def run_tests():
    """Run basic tests to verify setup."""
    if not os.path.exists('test_basic.py'):
        print("✗ test_basic.py not found")
        return False
    
    return run_command(
        [sys.executable, 'test_basic.py'],
        "Running basic tests"
    )


def main():
    """Main setup function."""
    print("Etymon Development Setup")
    print("=" * 40)
    
    # Check Python version
    if not check_python_version():
        return 1
    
    # Create necessary directories
    if not create_directories():
        print("\n✗ Directory creation failed")
        return 1
    
    # Install dependencies
    if not install_dependencies():
        print("\n✗ Dependency installation failed")
        print("\nTry installing dependencies manually:")
        print("  pip install -r requirements.txt")
        return 1
    
    # Run basic tests
    if not run_tests():
        print("\n✗ Basic tests failed")
        print("\nThere may be issues with your setup. Check error messages above.")
        return 1
    
    print("\n" + "=" * 40)
    print("✓ Setup completed successfully!")
    print("\nYou can now run Etymon:")
    print("  python main.py")
    print("\nOr run tests:")
    print("  python test_basic.py")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
