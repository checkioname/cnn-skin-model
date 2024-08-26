# Use the official Ubuntu as the base image
FROM ubuntu:latest

# Set environment variables to avoid interactive prompts during installation
ENV DEBIAN_FRONTEND=noninteractive

# Update the package repository and install essential packages
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    && apt-get clean

# Install required Python packages
RUN pip3 install numpy pandas torch torchvision tensorboard pillow matplotlib

# Set the working directory
WORKDIR /app

# Add your application code and scripts if needed
COPY . /app

# Start your application
# CMD [ "python", "your_script.py" ]

