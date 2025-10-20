# Stage 1: Build the virtual environment
FROM nvidia/cuda:12.1.1-devel-ubuntu22.04 AS builder

# Install Python, pip, and uv
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y python3.12 python3-pip python3.12-venv && \
    pip3 install uv && \
    rm -rf /var/lib/apt/lists/*

ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy
WORKDIR /app

# Copy project dependency files
COPY uv.lock pyproject.toml /app/

# Install torch with CUDA 12.1 support
RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Create a virtual environment and install dependencies using uv
RUN uv venv && \
    . .venv/bin/activate && \
    uv sync

# Stage 2: Create the final image
FROM nvidia/cuda:12.1.1-devel-ubuntu22.04
WORKDIR /app

# Install Python runtime
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y python3.12 && \
    rm -rf /var/lib/apt/lists/*

# Copy the virtual environment from the builder stage
COPY --from=builder /app/.venv ./.venv
# Copy the uv executable from the builder stage
COPY --from=builder /usr/local/bin/uv /usr/local/bin/uv

# Copy application code and model
COPY ragu-lm ./ragu-lm
COPY app.py .

# Set PATH to include the virtual environment's bin directory and uv
ENV PATH="/app/.venv/bin:/usr/local/bin:$PATH"
ENV NVIDIA_DRIVER_CAPABILITIES=all

EXPOSE 8000

# Run the application using uv
CMD ["uv", "run", "app.py"]