FROM python:3.9-slim

# Set the working directory
WORKDIR /app

# Copy necessary files into the Docker image
COPY requirements.txt ./
COPY app.py ./
COPY model.pkl ./
COPY mappings.json ./

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port and start the server
EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]

# docker build -t salary-predictor .
# docker run -p 8000:8000 salary-predictor
