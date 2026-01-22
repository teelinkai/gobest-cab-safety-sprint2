"""
Test Module for Mode Controller
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.core.mode_controller import ModeController
from src import config


def test_mode_controller_initialization():
    """Test that controller initializes correctly"""
    controller = ModeController()
    assert controller.current_mode == config.MODE_BATCH
    assert len(controller.session_history) == 0
    assert controller.current_prediction is None


def test_mode_switching():
    """Test mode switching functionality"""
    controller = ModeController()
    
    # Switch to realtime
    controller.set_mode(config.MODE_REALTIME)
    assert controller.get_mode() == config.MODE_REALTIME
    
    # Switch back to batch
    controller.set_mode(config.MODE_BATCH)
    assert controller.get_mode() == config.MODE_BATCH


def test_history_management():
    """Test history adding and retrieval"""
    controller = ModeController()
    
    # Add prediction to history
    prediction = {
        'mode': 'batch',
        'prediction': 'SAFE',
        'confidence': 0.85
    }
    controller.add_to_history(prediction)
    
    history = controller.get_history()
    assert len(history) == 1
    assert 'timestamp' in history[0]
    
    # Clear history
    controller.clear_history()
    assert len(controller.get_history()) == 0


if __name__ == "__main__":
    print("Running tests...")
    test_mode_controller_initialization()
    print("✓ Initialization test passed")
    
    test_mode_switching()
    print("✓ Mode switching test passed")
    
    test_history_management()
    print("✓ History management test passed")
    
    print("\nAll tests passed!")
