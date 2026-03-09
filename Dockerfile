FROM python:3.11-slim

WORKDIR /app


RUN apt-get update && apt-get install -y \
    build-essential git curl cmake \
    libglib2.0-0 libsm6 libxext6 libxrender1 \
    && rm -rf /var/lib/apt/lists/*


RUN pip install --no-cache-dir --upgrade pip setuptools wheel


RUN pip install --no-cache-dir \
    torch==2.3.0+cpu \
    torchvision==0.18.0+cpu \
    torchaudio==2.3.0 \
    -f https://download.pytorch.org/whl/cpu/torch_stable.html


RUN pip install --no-cache-dir torch-scatter -f https://data.pyg.org/whl/torch-2.3.0+cpu.html \
    && pip install --no-cache-dir torch-sparse -f https://data.pyg.org/whl/torch-2.3.0+cpu.html \
    && pip install --no-cache-dir torch-cluster -f https://data.pyg.org/whl/torch-2.3.0+cpu.html \
    && pip install --no-cache-dir torch-spline-conv -f https://data.pyg.org/whl/torch-2.3.0+cpu.html \
    && pip install --no-cache-dir torch-geometric -f https://data.pyg.org/whl/torch-2.3.0+cpu.html


COPY ./requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt


COPY . /app


EXPOSE 8000 8501


CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port 8000 & streamlit run app.py --server.port 8501 --server.address 0.0.0.0"]


