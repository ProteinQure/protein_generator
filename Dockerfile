FROM proteinqure/base-minimal AS build

RUN dnf install wget -y

ENV CONDA_DIR=/opt/conda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda

# Put conda in path so we can use conda activate
ENV PATH=$CONDA_DIR/bin:$PATH

WORKDIR /protein_generator
COPY environment.yml .

RUN --mount=type=cache,target=/opt/conda/pkgs conda env create -f environment.yml
RUN conda install -c conda-forge conda-pack
RUN conda-pack -n proteingenerator -o /tmp/env.tar && \
  mkdir /venv && cd /venv && tar -xf /tmp/env.tar && \
  rm /tmp/env.tar
COPY . .


FROM debian:bullseye-slim AS runtime

COPY --from=build /venv /venv
COPY --from=build /protein_generator /protein_generator

RUN ["/venv/bin/python", "-m", "pip", "install", "fastapi[standard]"]

CMD ["/venv/bin/python", "-m", "fastapi", "run", "/protein_generator/api.py", "--port", "80"]