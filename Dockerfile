# Use an Ubuntu base image
FROM ubuntu:23.10

# Set non-interactive timezone configuration
ENV TZ=UTC
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# Install system dependencies
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y \
    python3 \
    python3-pip \
    git \
    build-essential \
    cmake \
    ninja-build \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /usr/src/app

# Clone the StochasticTree repository with submodules
RUN git clone --recursive https://github.com/andrewherren/StochasticTree.git

# Checkout the specific commit
WORKDIR /usr/src/app/StochasticTree
RUN git checkout 5aff2ba68b33db479703ca8dd815f437feb66ea6

# Build the C++ components with detailed logging
RUN rm -rf build && mkdir build
RUN cmake -S . -B build -DCMAKE_VERBOSE_MAKEFILE:BOOL=ON
RUN cmake --build build

# Build and install the Python package from the StochasticTree
WORKDIR /usr/src/app/StochasticTree/python-package
RUN python3 setup.py build
RUN python3 setup.py install

# Return to the app directory
WORKDIR /usr/src/app

# Copy the current directory contents into the container
COPY . .

# Install Python dependencies from requirements.txt
# Exclude StochasticTree from requirements.txt if it's listed there
RUN pip install --no-cache-dir -r requirements.txt

# Make port 8000 available
EXPOSE 8000

# Run the FastAPI application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
