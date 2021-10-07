FROM ubuntu:18.04
# Copy Source
COPY mnist-dataset /exp/mnist-dataset
COPY requirements.txt /exp/requirements.txt
# Install Dependencies
RUN apt-get update && apt-get install -y python3 python3-pip
RUN pip3 install --no-cache-dir -r /exp/requirements.txt
# Set Entrypoint
RUN mkdir /exp/models
WORKDIR /exp
CMD ["python3", "./mnist-dataset/plot_graph.py"]
