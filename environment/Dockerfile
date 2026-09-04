# hash:sha256:73d1c8661ba989cf6d680f1e660fbb5348889ee7c9d33ea475414770dd59783e
ARG REGISTRY_HOST
FROM $REGISTRY_HOST/codeocean/swdb:Python3.12code-server4.133

ARG DEBIAN_FRONTEND=noninteractive
ARG GIT_ASKPASS
COPY git-ask-pass /

# No spaces around the `=`. Docker's legacy `ENV <key> <value>` form takes
# everything after the first space as the value, so the spaced version set this to
# "= 17cacea..." and the 2026-09-01 asset shipped that string as its version.
# Bump by hand whenever code/ changes -- nothing derives this, and a stale value
# makes the asset claim it was built from a commit it was not.
ENV SWDB_CODE_VERSION=eea6957cb014edf4c5efd5f122e012a8af97082e

COPY pyproject.toml /tmp/pyproject.toml
RUN uv pip install --system --no-cache -r /tmp/pyproject.toml