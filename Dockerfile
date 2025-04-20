FROM python:3.10-slim-bullseye

# Install system dependencies + Rust/Cargo
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    libssl-dev \
    pkg-config \
    && curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y \
    && echo 'source $HOME/.cargo/env' >> ~/.bashrc

# Ensure Rust is in PATH
ENV PATH="/root/.cargo/bin:${PATH}"

# Rest of your Dockerfile...
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
CMD ["gunicorn", "app:app", "-b", "0.0.0.0:5000"]
