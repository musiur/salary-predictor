FROM ubuntu:20.04

# Install dependencies
RUN apt-get update && apt-get install -y \
    curl \
    build-essential \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Download and install Ollama (modify with the correct URL if needed)
RUN curl -L https://ollama.com/download/ollama -o /usr/local/bin/ollama
RUN chmod +x /usr/local/bin/ollama

# Expose the default port
EXPOSE 11434

# Start the Ollama server
CMD ["ollama", "serve"]
