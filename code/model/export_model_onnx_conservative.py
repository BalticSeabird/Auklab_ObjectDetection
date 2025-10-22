#!/usr/bin/env python3
"""
Conservative TensorRT engine generation for stable batch processing
Optimized for dual RTX 4090 setup with minimal memory pressure
"""

import os
import torch
import logging
import tensorrt as trt
import numpy as np
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_conservative_engine():
    """Create TensorRT engine with extremely conservative memory settings for batch stability"""
    
    # Paths
    model_dir = Path("/home/jonas/Documents/vscode/Auklab_ObjectDetection/models")
    onnx_path = model_dir / "best.onnx"
    engine_path = model_dir / "best_conservative.engine"
    
    logger.info(f"🔧 Creating conservative TensorRT engine...")
    logger.info(f"📁 ONNX model: {onnx_path}")
    logger.info(f"📁 Engine output: {engine_path}")
    
    # Initialize TensorRT
    TRT_LOGGER = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(TRT_LOGGER)
    
    # Create network
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, TRT_LOGGER)
    
    # Parse ONNX model
    logger.info("📖 Parsing ONNX model...")
    with open(onnx_path, 'rb') as model:
        if not parser.parse(model.read()):
            logger.error("❌ Failed to parse ONNX model!")
            for error in range(parser.num_errors):
                logger.error(f"   {parser.get_error(error)}")
            return None
    
    # Create builder config with ULTRA conservative settings
    config = builder.create_builder_config()
    
    # Memory settings - extremely conservative (4GB instead of 8GB)
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 4 * 1024 * 1024 * 1024)  # 4GB
    
    # Optimization level - prioritize stability over speed
    config.set_builder_optimization_level(1)  # Lower optimization = more stable
    
    # Precision settings - FP16 for better memory efficiency
    if builder.platform_has_fast_fp16:
        logger.info("✅ Enabling FP16 precision for memory efficiency")
        config.set_flag(trt.BuilderFlag.FP16)
    
    # Disable aggressive optimizations that might cause instability
    config.set_flag(trt.BuilderFlag.DISABLE_TIMING_CACHE)
    
    # Configure dynamic batch sizes with smaller max batch
    input_tensor = network.get_input(0)
    input_tensor.shape = (-1, 3, 960, 960)  # Dynamic batch
    
    # Conservative batch range: 1-8 instead of 1-32
    profile = builder.create_optimization_profile()
    profile.set_shape("images", 
                     min=(1, 3, 960, 960),     # Min batch 1
                     opt=(4, 3, 960, 960),     # Optimal batch 4 (conservative)
                     max=(8, 3, 960, 960))     # Max batch 8 (much smaller than before)
    config.add_optimization_profile(profile)
    
    # Engine generation settings for stability
    config.set_flag(trt.BuilderFlag.OBEY_PRECISION_CONSTRAINTS)
    
    logger.info("🏗️  Building TensorRT engine with conservative settings...")
    logger.info(f"   📊 Batch range: 1-8 (optimal: 4)")
    logger.info(f"   💾 Workspace: 4GB")
    logger.info(f"   🎯 Optimization level: 1 (stability priority)")
    
    # Build engine
    engine = builder.build_serialized_network(network, config)
    
    if engine is None:
        logger.error("❌ Failed to build TensorRT engine!")
        return None
    
    # Save engine
    logger.info(f"💾 Saving engine to {engine_path}")
    with open(engine_path, 'wb') as f:
        f.write(engine)
    
    # Verify engine size
    engine_size = os.path.getsize(engine_path) / (1024 * 1024)
    logger.info(f"✅ Conservative engine created successfully!")
    logger.info(f"   📁 Size: {engine_size:.1f} MB")
    logger.info(f"   🎯 Optimized for batch sizes 1-8")
    logger.info(f"   💾 Memory footprint: Conservative (4GB workspace)")
    
    return engine_path

def verify_cuda_environment():
    """Verify CUDA environment before engine generation"""
    logger.info("🔍 Verifying CUDA environment...")
    
    # Check PyTorch CUDA
    logger.info(f"   🐍 PyTorch CUDA: {torch.version.cuda}")
    logger.info(f"   🔢 PyTorch version: {torch.__version__}")
    logger.info(f"   🎯 CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        logger.info(f"   📊 GPU count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            logger.info(f"   🎮 GPU {i}: {props.name} ({props.total_memory // 1024**3}GB)")
    
    # Check TensorRT version
    logger.info(f"   ⚡ TensorRT version: {trt.__version__}")
    
    return True

if __name__ == "__main__":
    try:
        # Verify environment
        verify_cuda_environment()
        
        # Create conservative engine
        engine_path = create_conservative_engine()
        
        if engine_path:
            logger.info("🎉 Conservative TensorRT engine generation completed!")
            logger.info("   This engine prioritizes stability over maximum performance")
            logger.info("   Recommended for reliable batch processing up to batch size 8")
        else:
            logger.error("❌ Engine generation failed!")
            
    except Exception as e:
        logger.error(f"❌ Error during engine generation: {e}")
        import traceback
        traceback.print_exc()