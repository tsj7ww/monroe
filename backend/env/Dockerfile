FROM python:3.11-slim

ARG ENVIRONMENT
ENV ENVIRONMENT=${ENVIRONMENT}

WORKDIR /workspace

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    # Bash
    bash \
    bash-completion \
    # System
    build-essential \
    curl \
    git \
    libpq-dev \
    # Tools
    gawk \
    sed \
    grep \
    ripgrep \
    parallel \
    jq \
    xmlstarlet \
    htop \
    glances \
    ncdu \
    netcat-openbsd \
    socat \
    nmap \
    tmux \
    vim \
    tree \
    pigz \
    pbzip2 \
    csvkit \
    datamash \
    bc \
    git-lfs \
    # Stan packages
    # wget \
    # cmake \
    # gfortran \
    # libopenblas-dev \
    # liblapack-dev \
    && rm -rf /var/lib/apt/lists/*

# Install pipenv and virtualenv
RUN pip install --no-cache-dir pipenv virtualenv
# Create a virtual environment
RUN python -m virtualenv /opt/venv/${ENVIRONMENT}
# Copy Pipfile
COPY Pipfile .
##################
### Cold Start ###
# RUN pipenv lock
### Warm Start ###
COPY Pipfile.lock .
##################
# Generate requirements.txt from Pipfile using pipenv
RUN pipenv requirements > requirements.txt
# Install packages in the virtual environment
RUN . /opt/venv/${ENVIRONMENT}/bin/activate && \
    pip install --no-cache-dir -r requirements.txt && \
    python -m ipykernel install --user --name=${ENVIRONMENT} --display-name=${ENVIRONMENT}

# Set up bash environment
COPY bashrc /root/.bashrc
ENV SHELL=/bin/bash
ENV PATH="/opt/venv/${ENVIRONMENT}/bin:$PATH"

# # Install CmdStan
# RUN mkdir -p ${STAN_HOME} \
#     && wget https://github.com/stan-dev/cmdstan/releases/download/v2.32.2/cmdstan-2.32.2.tar.gz \
#     && tar -xzf cmdstan-2.32.2.tar.gz -C ${STAN_HOME} --strip-components=1 \
#     && rm cmdstan-2.32.2.tar.gz \
#     && cd ${STAN_HOME} \
#     && make build
# # Set up CmdStan for PyStan
# RUN python -m cmdstanpy.install_cmdstan

# Expose ports: Jupyter
# Ray dashboard + head node
# Dask scheduler + dashboard
EXPOSE 8888 \
    8265 6379 \
    8786 8787

# Set up an entrypoint script
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]