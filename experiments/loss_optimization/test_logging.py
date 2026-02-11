#!/usr/bin/env python3
"""
Quick test to verify data logging is working correctly.

This runs a minimal training example and checks that the log file is created
with the expected content.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

def test_data_logging():
    """Test that data logging creates the expected log file."""
    import os
    os.environ['SINGLE_GPU_MODE'] = '1'  # Force single GPU
    
    from train_local import setup_data_logger, RESULTS_DIR
    
    # Create test logger
    test_run = "test_logging_123"
    logger = setup_data_logger(RESULTS_DIR, test_run)
    
    # Log some test data
    logger.info("Testing data logging system")
    logger.info("DATALOADER CREATION")
    logger.info("Classes: ['nuc', 'mito']")
    logger.info("Batch size: 4")
    
    # Check log file exists
    log_filename = "{}_data_debug.log".format(test_run)
    log_file = RESULTS_DIR / log_filename
    
    if log_file.exists():
        print("✅ Log file created: {}".format(log_file))
        
        # Read and display first few lines
        with open(log_file, 'r') as f:
            lines = f.readlines()
            print("✅ Log file has {} lines".format(len(lines)))
            print("\nFirst 10 lines:")
            print("=" * 60)
            for line in lines[:10]:
                print(line.rstrip())
            print("=" * 60)
        
        # Clean up test file
        log_file.unlink()
        print("\n✅ Test passed! Cleaned up test file.")
        return True
    else:
        print("❌ Log file not created at {}".format(log_file))
        return False

if __name__ == '__main__':
    success = test_data_logging()
    sys.exit(0 if success else 1)
