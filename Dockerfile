FROM python:3.10-slim

WORKDIR /app

# CPU-only PyTorch: the default Linux wheel bundles CUDA (~2 GB), which
# is pointless here. Installed first so requirements.txt sees it satisfied.
RUN pip install --no-cache-dir torch torchvision \
    --extra-index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN pip install --no-cache-dir -e .

CMD ["pytest"]
