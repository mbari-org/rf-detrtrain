# AWS SageMaker Training Container for RF-DETR
# Based on PyTorch deep learning container

# Use AWS Deep Learning Container as base (use us-west-2 region)
# AWS DLC images are region-specific
# Using PyTorch 2.1 for better torch.compiler support
FROM 763104351884.dkr.ecr.us-west-2.amazonaws.com/pytorch-training:2.8.0-gpu-py312-cu129-ubuntu22.04-sagemaker-v1.7

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    PATH="/opt/ml/code:${PATH}"

# Install additional system dependencies for OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /opt/ml/code

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip setuptools wheel


# # Install rfdetr (this will install its dependencies and override if needed)
RUN pip install --no-cache-dir --upgrade rfdetr

# # Verify installations and print versions for debugging
RUN python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}'); print(f'Has compiler: {hasattr(torch, \"compiler\")}')" && \
    python -c "import transformers; print(f'Transformers: {transformers.__version__}')" && \
    python -c "import accelerate; print(f'Accelerate: {accelerate.__version__}')" && \
    python -c "import rfdetr; print(f'RFDETR installed successfully')" || echo "RFDETR check completed"

# Copy training script
COPY src/train_rfdetr_aws.py /opt/ml/code/train.py

# Make script executable
RUN chmod +x /opt/ml/code/train.py

# SageMaker runs the container with:
# docker run image train [--hyperparameter1 value1 --hyperparameter2 value2 ...]
ENV SAGEMAKER_PROGRAM train.py

# Set up SageMaker training directories
RUN mkdir -p /opt/ml/model && \
    mkdir -p /opt/ml/output && \
    mkdir -p /opt/ml/input/data/training && \
    mkdir -p /opt/ml/input/config

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import torch; import rfdetr" || exit 1

# Default command (will be overridden by SageMaker)
ENTRYPOINT ["python", "/opt/ml/code/train.py"]
