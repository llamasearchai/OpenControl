#!/usr/bin/env python3
"""
Simplified System Test for OpenControl.

This script tests basic functionality of the OpenControl system
to ensure core components work correctly.

Author: Nik Jois <nikjois@llamasearch.ai>
"""

import sys
from pathlib import Path
import logging

# Add the project root to the path
sys.path.insert(0, str(Path(__file__).parent))

def test_imports():
    """Test that all core modules can be imported."""
    print("Testing imports...")
    
    try:
        from opencontrol.cli.commands import OpenControlConfig
        print("✓ CLI commands imported")
        
        from opencontrol.core.world_model import OpenControlWorldModel
        print("✓ World model imported")
        
        from opencontrol.data.dataset_manager import MultiModalDatasetManager
        print("✓ Data manager imported")
        
        from opencontrol.utils.logging_utils import setup_logging
        print("✓ Logging utils imported")
        
        return True
    except Exception as e:
        print(f"✗ Import failed: {e}")
        return False

def test_configuration():
    """Test configuration system."""
    print("\nTesting configuration...")
    
    try:
        from opencontrol.cli.commands import OpenControlConfig
        
        # Create default configuration
        config = OpenControlConfig()
        print("✓ Default configuration created")
        
        # Check basic attributes
        assert hasattr(config, 'model')
        assert hasattr(config, 'training')
        assert hasattr(config, 'control')
        print("✓ Configuration has required sections")
        
        return True
    except Exception as e:
        print(f"✗ Configuration test failed: {e}")
        return False

def test_world_model():
    """Test world model creation."""
    print("\nTesting world model...")
    
    try:
        from opencontrol.cli.commands import OpenControlConfig
        from opencontrol.core.world_model import OpenControlWorldModel
        
        # Create configuration
        config = OpenControlConfig()
        
        # Create world model
        model = OpenControlWorldModel(config)
        print("✓ World model created")
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        print(f"✓ Model has {total_params:,} parameters")
        
        return True
    except Exception as e:
        print(f"✗ World model test failed: {e}")
        return False

def test_data_manager():
    """Test data management system."""
    print("\nTesting data manager...")
    
    try:
        from opencontrol.cli.commands import OpenControlConfig
        from opencontrol.data.dataset_manager import MultiModalDatasetManager
        
        # Create configuration
        config = OpenControlConfig()
        
        # Create data manager
        data_manager = MultiModalDatasetManager(config)
        print("✓ Data manager created")
        
        # Initialize with mock data
        import asyncio
        asyncio.run(data_manager.initialize())
        print("✓ Data manager initialized")
        
        # Test data loaders
        train_loader = data_manager.get_train_loader()
        print(f"✓ Train loader created with {len(train_loader)} batches")
        
        return True
    except Exception as e:
        print(f"✗ Data manager test failed: {e}")
        return False

def test_monitoring():
    """Test monitoring system."""
    print("\nTesting monitoring...")
    
    try:
        from opencontrol.cli.commands import OpenControlConfig
        from opencontrol.deployment.monitoring import ProductionMonitor
        
        # Create configuration
        config = OpenControlConfig()
        
        # Create monitor
        monitor = ProductionMonitor(config)
        print("✓ Monitor created")
        
        # Log some metrics
        monitor.log_prediction_metrics(0.05, 1)
        monitor.log_control_metrics(0.02, 1.5, 'safe')
        print("✓ Metrics logged")
        
        # Get metrics
        metrics = monitor.get_metrics()
        print(f"✓ Metrics retrieved: {list(metrics.keys())}")
        
        return True
    except Exception as e:
        print(f"✗ Monitoring test failed: {e}")
        return False

def test_cli():
    """Test CLI functionality."""
    print("\nTesting CLI...")
    
    try:
        # Test CLI import and basic functionality
        import subprocess
        result = subprocess.run([sys.executable, "-c", 
                               "from opencontrol.cli.main import OpenControlCLI; print('CLI imported successfully')"],
                               capture_output=True, text=True, timeout=10)
        
        if result.returncode == 0:
            print("✓ CLI module can be imported")
            return True
        else:
            print(f"✗ CLI test failed: {result.stderr}")
            return False
    except Exception as e:
        print(f"✗ CLI test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("OpenControl Simplified System Test")
    print("Author: Nik Jois <nikjois@llamasearch.ai>")
    print("=" * 50)
    
    tests = [
        ("Imports", test_imports),
        ("Configuration", test_configuration), 
        ("World Model", test_world_model),
        ("Data Manager", test_data_manager),
        ("Monitoring", test_monitoring),
        ("CLI", test_cli)
    ]
    
    results = {}
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"✗ {test_name} test crashed: {e}")
            results[test_name] = False
    
    # Summary
    print("\n" + "=" * 50)
    print("TEST RESULTS SUMMARY")
    print("=" * 50)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results.items():
        status = "PASS" if result else "FAIL"
        emoji = "✓" if result else "✗"
        print(f"{emoji} {test_name:20} : {status}")
        if result:
            passed += 1
    
    print("=" * 50)
    print(f"OVERALL RESULT: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED! OpenControl system is working correctly.")
        return True
    else:
        print(f"❌ {total - passed} tests failed.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 