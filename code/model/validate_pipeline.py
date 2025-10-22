#!/usr/bin/env python3
"""
Pipeline Validation Script
Test that the production pipeline works with the clean TensorRT engine
"""

import os
import sys
import time
import subprocess

def check_dependencies():
    """Check if all required packages are available"""
    print("🔍 Checking dependencies...")
    
    try:
        import tensorrt as trt
        print(f"✅ TensorRT: {trt.__version__}")
    except ImportError:
        print("❌ TensorRT not available")
        return False
    
    try:
        import pycuda.driver as cuda
        import pycuda.autoinit
        print(f"✅ PyCUDA: {cuda.Device.count()} devices")
    except ImportError:
        print("❌ PyCUDA not available")
        return False
    
    try:
        import torch
        print(f"✅ PyTorch: {torch.__version__}")
        print(f"   CUDA available: {torch.cuda.is_available()}")
        print(f"   GPUs: {torch.cuda.device_count()}")
    except ImportError:
        print("❌ PyTorch not available")
        return False
    
    try:
        import av
        print(f"✅ PyAV available")
    except ImportError:
        print("❌ PyAV not available")
        return False
    
    return True

def check_files():
    """Check if required files exist"""
    print("\n📁 Checking required files...")
    
    # Check clean engine
    clean_engine = "models/auklab_model_xlarge_combined_4564_v1_clean.trt"
    if os.path.exists(clean_engine):
        size = os.path.getsize(clean_engine) / 1024 / 1024
        print(f"✅ Clean engine: {clean_engine} ({size:.1f} MB)")
    else:
        print(f"❌ Clean engine not found: {clean_engine}")
        return False
    
    # Check production script
    prod_script = "code/model/production_batch_inference.py"
    if os.path.exists(prod_script):
        print(f"✅ Production script: {prod_script}")
    else:
        print(f"❌ Production script not found: {prod_script}")
        return False
    
    # Check test video
    test_video = "vid/input.mp4"
    if os.path.exists(test_video):
        size = os.path.getsize(test_video) / 1024 / 1024
        print(f"✅ Test video: {test_video} ({size:.1f} MB)")
    else:
        print(f"❌ Test video not found: {test_video}")
        return False
    
    return True

def test_engine_loading():
    """Test that the clean engine loads properly"""
    print("\n🔧 Testing clean engine loading...")
    
    try:
        import tensorrt as trt
        import pycuda.driver as cuda
        import pycuda.autoinit
        
        # Load the clean engine
        engine_path = "models/auklab_model_xlarge_combined_4564_v1_clean.trt"
        TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
        
        with open(engine_path, 'rb') as f:
            engine_data = f.read()
        
        runtime = trt.Runtime(TRT_LOGGER)
        engine = runtime.deserialize_cuda_engine(engine_data)
        
        if engine is None:
            print("❌ Failed to deserialize engine")
            return False
        
        print("✅ Engine loaded successfully")
        
        # Create execution context
        context = engine.create_execution_context()
        print("✅ Execution context created")
        
        # Check GPU memory requirements
        try:
            memory_size = engine.get_device_memory_size_v2() / 1024 / 1024
        except AttributeError:
            # Fallback for older TensorRT versions
            memory_size = engine.device_memory_size / 1024 / 1024
        print(f"📊 GPU memory required: {memory_size:.1f} MB")
        
        return True
        
    except Exception as e:
        print(f"❌ Engine loading failed: {e}")
        return False

def test_quick_inference():
    """Test a quick inference run with the production script"""
    print("\n🚀 Testing production pipeline...")
    
    try:
        # Run the production script with a short test
        cmd = [
            "python3", "code/model/production_batch_inference.py",
            "vid/input.mp4",
            "--batch-size", "8",  # Use optimal batch size from our testing
            "--frame-skip", "100",  # Process only every 100th frame for quick test
        ]
        
        print(f"Running: {' '.join(cmd)}")
        print("⏱️ This should take less than 30 seconds...")
        
        start_time = time.time()
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        end_time = time.time()
        
        print(f"⏱️ Execution time: {end_time - start_time:.1f} seconds")
        
        if result.returncode == 0:
            print("✅ Production pipeline executed successfully!")
            
            # Show some output
            if result.stdout:
                lines = result.stdout.strip().split('\n')
                print("📊 Pipeline output:")
                for line in lines[-10:]:  # Show last 10 lines
                    print(f"   {line}")
            
            return True
        else:
            print("❌ Production pipeline failed!")
            print("STDOUT:", result.stdout)
            print("STDERR:", result.stderr)
            return False
            
    except subprocess.TimeoutExpired:
        print("❌ Test timed out (>60 seconds)")
        return False
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

def main():
    """Run all validation tests"""
    print("🧪 PIPELINE VALIDATION")
    print("=" * 50)
    
    all_passed = True
    
    # Test 1: Dependencies
    if not check_dependencies():
        all_passed = False
    
    # Test 2: Files
    if not check_files():
        all_passed = False
    
    # Test 3: Engine loading
    if not test_engine_loading():
        all_passed = False
    
    # Test 4: Quick inference (only if previous tests pass)
    if all_passed:
        if not test_quick_inference():
            all_passed = False
    else:
        print("\n⚠️ Skipping inference test due to previous failures")
    
    # Summary
    print("\n" + "=" * 50)
    if all_passed:
        print("🎉 ALL TESTS PASSED!")
        print("✅ Production pipeline is ready for use")
        print("✅ Clean TensorRT engine is working")
        print("✅ Expected performance: 94.7+ images/second")
        print("\n🚀 Ready for multi-GPU implementation!")
    else:
        print("❌ SOME TESTS FAILED")
        print("⚠️ Check the errors above before proceeding")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)