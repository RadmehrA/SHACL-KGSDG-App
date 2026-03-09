# # Use an official Python runtime as a parent image
# FROM python:3.9-slim

# # Set the working directory in the container
# WORKDIR /app

# # Copy the requirements.txt first to leverage Docker's cache for dependencies
# COPY requirements.txt /app/

# # Install dependencies for both FastAPI and Streamlit
# RUN pip install --no-cache-dir -r requirements.txt

# # Now copy the rest of the application files
# COPY . /app

# # Expose the ports the apps will run on
# EXPOSE 8000 8501

# # Command to run both FastAPI and Streamlit using a shell script
# CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port 8000 & streamlit run app.py --server.port 8501"]


# # Use an official Python runtime as a parent image
# FROM python:3.9-slim

# # Set the working directory in the container
# WORKDIR /app

# # Copy the requirements.txt first to leverage Docker's cache for dependencies
# COPY requirements.txt /app/

# # Install dependencies for both FastAPI and Streamlit
# RUN pip install --no-cache-dir -r requirements.txt

# # Now copy the rest of the application files
# COPY . /app

# # Expose the ports the apps will run on
# EXPOSE 8000 8501

# # Command to run both FastAPI and Streamlit using a shell script
# CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port 8000 & streamlit run app.py --server.port 8501"]


# FROM python:3.10-slim

# WORKDIR /app

# # Install system dependencies
# RUN apt-get update && apt-get install -y \
#     libglib2.0-0 libsm6 libxext6 libxrender1 \
#     build-essential git curl cmake \
#     && rm -rf /var/lib/apt/lists/*

# # Copy requirements
# COPY ./requirements.txt /app/requirements.txt

# # Upgrade pip
# RUN pip install --no-cache-dir --upgrade pip

# # Install CPU PyTorch + torchvision
# RUN pip install --no-cache-dir torch==2.1.0+cpu torchvision==0.16.0+cpu -f https://download.pytorch.org/whl/cpu/torch_stable.html

# # Install PyTorch Geometric CPU wheels
# RUN pip install --no-cache-dir torch-scatter -f https://data.pyg.org/whl/torch-2.1.0+cpu.html \
#     && pip install --no-cache-dir torch-sparse -f https://data.pyg.org/whl/torch-2.1.0+cpu.html \
#     && pip install --no-cache-dir torch-cluster -f https://data.pyg.org/whl/torch-2.1.0+cpu.html \
#     && pip install --no-cache-dir torch-spline-conv -f https://data.pyg.org/whl/torch-2.1.0+cpu.html \
#     && pip install --no-cache-dir torch-geometric -f https://data.pyg.org/whl/torch-2.1.0+cpu.html

# # Install remaining dependencies
# RUN pip install --no-cache-dir -r requirements.txt

# # Copy the rest of the application (both backend + frontend)
# COPY . /app

# # Expose ports for both apps
# EXPOSE 8000 8501

# # Run both FastAPI and Streamlit together
# CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port 8000 & streamlit run app.py --server.port 8501 --server.address 0.0.0.0"]


# --------------------------
# Base Image
# --------------------------
FROM python:3.11-slim

WORKDIR /app

# --------------------------
# System Dependencies
# --------------------------
RUN apt-get update && apt-get install -y \
    build-essential git curl cmake \
    libglib2.0-0 libsm6 libxext6 libxrender1 \
    && rm -rf /var/lib/apt/lists/*

# --------------------------
# Upgrade pip
# --------------------------
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# --------------------------
# PyTorch + TorchVision + Torchaudio (CPU, compatible with Mac M2)
# --------------------------
RUN pip install --no-cache-dir \
    torch==2.3.0+cpu \
    torchvision==0.18.0+cpu \
    torchaudio==2.3.0 \
    -f https://download.pytorch.org/whl/cpu/torch_stable.html

# --------------------------
# PyTorch Geometric CPU wheels for 2.3.0
# --------------------------
RUN pip install --no-cache-dir torch-scatter -f https://data.pyg.org/whl/torch-2.3.0+cpu.html \
    && pip install --no-cache-dir torch-sparse -f https://data.pyg.org/whl/torch-2.3.0+cpu.html \
    && pip install --no-cache-dir torch-cluster -f https://data.pyg.org/whl/torch-2.3.0+cpu.html \
    && pip install --no-cache-dir torch-spline-conv -f https://data.pyg.org/whl/torch-2.3.0+cpu.html \
    && pip install --no-cache-dir torch-geometric -f https://data.pyg.org/whl/torch-2.3.0+cpu.html

# --------------------------
# Install Remaining Python Dependencies
# --------------------------
# Make sure torch/torchvision/torchaudio are NOT in requirements.txt
COPY ./requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# --------------------------
# Copy Application
# --------------------------
COPY . /app

# --------------------------
# Expose Ports (FastAPI + Streamlit)
# --------------------------
EXPOSE 8000 8501

# --------------------------
# Run both FastAPI and Streamlit
# --------------------------
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port 8000 & streamlit run app.py --server.port 8501 --server.address 0.0.0.0"]


