FROM public.ecr.aws/docker/library/python:3.11.11-slim-bookworm AS builder

RUN apt-get update && \
    apt-get install -y \
        g++ \
        make \
        cmake \
        pkg-config \
        unzip \
        libcurl4-openssl-dev \
        build-essential \
        libopenblas-dev \
        git && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN python3 -m pip install --upgrade pip

# Optional: CMake args for BLAS for llama-cpp-python installation
ENV CMAKE_ARGS="-DGGML_BLAS=ON -DGGML_BLAS_VENDOR=OpenBLAS"

WORKDIR /src

COPY requirements_aws.txt .

RUN pip install torch==2.5.1+cpu --target=/install --index-url https://download.pytorch.org/whl/cpu \
&& pip install --no-cache-dir --target=/install sentence-transformers==4.1.0 --no-deps \
&& pip install --no-cache-dir --target=/install span-marker==1.7.0 --no-deps \
&& pip install --no-cache-dir --target=/install langchain-huggingface==0.1.2 --no-deps \
&& pip install --no-cache-dir --target=/install keybert==0.9.0 --no-deps \
&& pip install --no-cache-dir --target=/install -r requirements_aws.txt

# Stage 2: Final runtime image
FROM public.ecr.aws/docker/library/python:3.11.11-slim-bookworm

RUN apt-get update && \
    apt-get install -y \
    libopenblas0 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Set up a new user named "user" with user ID 1000
RUN useradd -m -u 1000 user

# Create required directories
RUN mkdir -p /home/user/app/{output,input,tld,logs,usage,feedback,config} \
    && chown -R user:user /home/user/app

# Copy installed packages from builder stage
COPY --from=builder /install /usr/local/lib/python3.11/site-packages/

# Switch to the "user" user
USER user

# Set home to the user's home directory
ENV APP_HOME=/home/user

ENV PATH=$APP_HOME/.local/bin:$PATH \
    PYTHONPATH=$APP_HOME/app \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    GRADIO_ALLOW_FLAGGING=never \
    GRADIO_NUM_PORTS=1 \
    GRADIO_SERVER_NAME=0.0.0.0 \
    GRADIO_SERVER_PORT=7860 \
    GRADIO_ANALYTICS_ENABLED=False \
    TLDEXTRACT_CACHE=$APP_HOME/app/tld/.tld_set_snapshot \
    SYSTEM=spaces \
	LLAMA_CUBLAS=0
 
# Set the working directory to the user's home directory
WORKDIR $APP_HOME/app

# Copy the current directory contents into the container at $HOME/app setting the owner to the user
COPY --chown=user . $APP_HOME/app

# Ensure permissions are really user:user again after copying
RUN chown -R user:user $APP_HOME/app && chmod -R u+rwX $APP_HOME/app

CMD ["python", "app.py"]