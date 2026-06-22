FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

RUN mkdir -p /opt/app /workdir
WORKDIR /opt/app

RUN python -m pip install -U pip && python -m pip install pip-tools

# Install the requirements
COPY requirements.txt /opt/app/
RUN python -m pip install -r requirements.txt

# Copy training scripts
COPY scripts/run_ner_strided.py /opt/app/
COPY training/ /opt/app/training/
COPY evaluation/ /opt/app/evaluation/
