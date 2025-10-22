#!/usr/bin/env python3
"""
Test DALI Installation and Basic Functionality
Verify that NVIDIA DALI can be installed and works with your system
"""

import sys
import subprocess
import os

def check_cuda_version():
    """Check CUDA version for DALI compatibility"""
    print("🔍 Checking CUDA environment...")
    
    try:
        import pycuda.driver as cuda
        import pycuda.autoinit
        
        # Get CUDA version
        cuda_version = cuda.get_version()
        driver_version = cuda.get_driver_version()
        
        print(f"   CUDA Runtime version: {cuda_version}")
        print(f"   CUDA Driver version: {driver_version}")
        
        # Get GPU info
        device = cuda.Device(0)
        print(f"   GPU: {device.name()}")
        print(f"   Compute capability: {device.compute_capability()}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ CUDA check failed: {e}")
        return False

def install_dali():
    """Install NVIDIA DALI with appropriate CUDA version"""
    print("\n📦 Installing NVIDIA DALI...")
    
    # Determine CUDA version for DALI
    # DALI supports CUDA 11.0, 11.2, 11.8, 12.0, 12.2, 12.4
    cuda_versions = {
        (11, 0): "cuda110",
        (11, 2): "cuda112", 
        (11, 8): "cuda118",
        (12, 0): "cuda120",
        (12, 2): "cuda122",
        (12, 4): "cuda124"
    }
    
    # Default to CUDA 12.4 (most recent supported)
    cuda_suffix = "cuda124"
    
    try:
        import pycuda.driver as cuda
        cuda_version = cuda.get_version()
        major, minor = cuda_version[0], cuda_version[1]
        
        if (major, minor) in cuda_versions:
            cuda_suffix = cuda_versions[(major, minor)]
            print(f"   Detected CUDA {major}.{minor}, using {cuda_suffix}")
        else:
            print(f"   CUDA {major}.{minor} detected, using default {cuda_suffix}")
            
    except:
        print(f"   Could not detect CUDA version, using default {cuda_suffix}")
    
    # Install command
    install_cmd = [
        sys.executable, "-m", "pip", "install", 
        "--extra-index-url", "https://developer.download.nvidia.com/compute/redist",
        "--upgrade", f"nvidia-dali-{cuda_suffix}"
    ]
    
    print(f"   Running: {' '.join(install_cmd)}")
    
    try:
        result = subprocess.run(install_cmd, capture_output=True, text=True, timeout=300)
        
        if result.returncode == 0:
            print("   ✅ DALI installation completed!")
            return True
        else:
            print(f"   ❌ Installation failed:")
            print(f"   stdout: {result.stdout}")
            print(f"   stderr: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print("   ❌ Installation timed out")
        return False
    except Exception as e:
        print(f"   ❌ Installation error: {e}")
        return False

def test_dali_import():
    """Test if DALI can be imported"""
    print("\n🧪 Testing DALI import...")
    
    try:
        import nvidia.dali as dali
        print(f"   ✅ DALI version: {dali.__version__}")
        
        from nvidia.dali import pipeline_def
        import nvidia.dali.fn as fn
        from nvidia.dali.plugin.pytorch import DALIGenericIterator
        from nvidia.dali import types
        
        print("   ✅ All DALI modules imported successfully")
        return True
        
    except ImportError as e:
        print(f"   ❌ DALI import failed: {e}")
        return False

def test_dali_basic_pipeline():
    """Test basic DALI functionality"""
    print("\n🔧 Testing basic DALI pipeline...")
    
    try:
        import nvidia.dali as dali
        from nvidia.dali import pipeline_def
        import nvidia.dali.fn as fn
        from nvidia.dali.plugin.pytorch import DALIGenericIterator
        from nvidia.dali import types
        import numpy as np
        
        @pipeline_def
        def simple_pipeline():
            # Create dummy data
            data = fn.constant(fdata=np.random.randn(3, 224, 224).astype(np.float32), device="gpu")
            # Simple resize operation
            resized = fn.resize(data, device="gpu", size=[256, 256])
            return resized
        
        # Build pipeline
        pipe = simple_pipeline(batch_size=2, num_threads=1, device_id=0)
        pipe.build()
        
        # Test iterator
        iterator = DALIGenericIterator([pipe], output_map=["data"], last_batch_policy="fill")
        
        # Get one batch
        batch = next(iterator)
        data = batch[0]["data"]
        
        print(f"   ✅ Pipeline test successful!")
        print(f"   ✅ Output shape: {data.shape}")
        print(f"   ✅ Output type: {type(data)}")
        
        # Cleanup
        del iterator
        del pipe
        
        return True
        
    except Exception as e:
        print(f"   ❌ Pipeline test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_dali_video_ops():
    """Test DALI video operations availability"""
    print("\n📹 Testing DALI video operations...")
    
    try:
        import nvidia.dali as dali
        import nvidia.dali.fn as fn
        
        # Check if video reader is available
        if hasattr(fn.readers, 'video'):
            print("   ✅ Video reader available")
        else:
            print("   ❌ Video reader not available")
            return False
        
        # Check video-related operations
        video_ops = ['resize', 'cast', 'transpose', 'normalize']
        for op in video_ops:
            if hasattr(fn, op):
                print(f"   ✅ {op} operation available")
            else:
                print(f"   ❌ {op} operation not available")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Video ops test failed: {e}")
        return False

def main():
    """Main installation and testing workflow"""
    print("🚀 NVIDIA DALI Installation and Testing")
    print("=" * 60)
    
    # Step 1: Check CUDA
    if not check_cuda_version():
        print("\n❌ CUDA environment issues detected. Please fix CUDA installation first.")
        return False
    
    # Step 2: Try importing DALI (might already be installed)
    if test_dali_import():
        print("\n✅ DALI already installed and working!")
    else:
        print("\n📦 DALI not found, attempting installation...")
        
        # Step 3: Install DALI
        if not install_dali():
            print("\n❌ DALI installation failed. Please check error messages above.")
            return False
        
        # Step 4: Test import after installation
        if not test_dali_import():
            print("\n❌ DALI installation completed but import still fails.")
            return False
    
    # Step 5: Test basic functionality
    if not test_dali_basic_pipeline():
        print("\n❌ DALI basic functionality test failed.")
        return False
    
    # Step 6: Test video operations
    if not test_dali_video_ops():
        print("\n❌ DALI video operations not available.")
        return False
    
    print("\n🎉 SUCCESS!")
    print("✅ NVIDIA DALI is properly installed and functional")
    print("✅ Ready for GPU-accelerated video decoding")
    print("\nNext steps:")
    print("1. Test with: python3 code/model/production_batch_inference_dali.py <video_file>")
    print("2. Benchmark with: python3 code/model/benchmark_decoding.py <video_file>")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)