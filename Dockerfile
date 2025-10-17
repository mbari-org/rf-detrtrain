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

# Copy src module and training script
COPY src/__init__.py /opt/ml/code/src/__init__.py
COPY src/train_rfdetr_aws.py /opt/ml/code/train.py
COPY src/main.py /opt/ml/code/main.py
COPY src/main.sh /opt/ml/code/main.sh

# Make script executable
RUN chmod +x /opt/ml/code/train.py
RUN chmod +x /opt/ml/code/main.sh

# Make train.py the default entrypoint for SageMaker.
# This can be overridden in the Session
ENV SAGEMAKER_PROGRAM train.py

# Set up SageMaker training directories
RUN mkdir -p /opt/ml/model && \
    mkdir -p /opt/ml/output && \
    mkdir -p /opt/ml/input/data/training && \
    mkdir -p /opt/ml/input/config

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import torch; import rfdetr" || exit 1

# SageMaker will use the SAGEMAKER_PROGRAM env var to run the training script
# No ENTRYPOINT needed - the base DLC image handles execution
