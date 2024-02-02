# Use the official Ubuntu 18.04 image
FROM ubuntu:18.04

# Install essential packages
RUN apt-get update && \
    apt-get install -y \
    python2.7 \
    python-pip \
    && rm -rf /var/lib/apt/lists/*

# Set the default Python version
# RUN ln -s /usr/bin/python2.7 /usr/bin/python
RUN /usr/bin/python2.7 --version
RUN /usr/bin/python2.7 -m pip install --upgrade pip

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container at /app
COPY requirements.txt /app/

# Install any dependencies specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the container at /app
COPY . /app

# Specify the default command to run on container start
CMD ["python", "your_script.py"]