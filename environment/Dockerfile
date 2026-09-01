# hash:sha256:73d1c8661ba989cf6d680f1e660fbb5348889ee7c9d33ea475414770dd59783e
ARG REGISTRY_HOST
FROM $REGISTRY_HOST/codeocean/swdb:Python3.12code-server4.133

ARG DEBIAN_FRONTEND=noninteractive
ARG GIT_ASKPASS
COPY git-ask-pass /

ENV SWDB_CODE_VERSION = 17cacea5a61c6b596324d6911a879b15f3ed98c4

COPY pyproject.toml /tmp/pyproject.toml
RUN uv pip install --system --no-cache -r /tmp/pyproject.toml