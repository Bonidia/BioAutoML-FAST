ARG BASE_IMAGE=bioautoml-fast:local
FROM ${BASE_IMAGE}
COPY --chown=1001:1001 example/ /app/bundled-example/
