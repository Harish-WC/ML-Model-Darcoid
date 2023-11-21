# Use an official Python runtime as a parent image
FROM python:3.9-slim

ENV PYTHONUNBUFFERED=TRUE \
    PYTHONDONTWRITEBYTECODE=TRUE \
    PATH="/home/model-server:${PATH}" \
    JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64

# Update the package repository and install required packages
RUN apt-get update \
    && apt-get install -y --no-install-recommends wget python3-setuptools nginx ca-certificates default-jre \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Create necessary directories and set permissions
RUN mkdir -p /opt/ml/model \
    && mkdir -p /home/model-server \
    && chown -R nobody:nogroup /home/model-server

# Set the working directory
WORKDIR /home/model-server

# Copy relevant files
COPY --chown=nobody:nogroup requirements.txt .
COPY --chown=nobody:nogroup app.py .
COPY --chown=nobody:nogroup model_handler.py .
COPY --chown=nobody:nogroup serve .
COPY --chown=nobody:nogroup wsgi.py .
COPY --chown=nobody:nogroup nginx.conf .

# Install Python packages from requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Switch to non-root user
USER nobody

# Start the server
CMD ["./serve"]
