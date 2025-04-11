# Use Debian as the base image
FROM python:3.11-slim-bookworm

# Install necessary system packages
RUN apt-get update && apt-get install curl --no-install-recommends -y \
    build-essential && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Download the latest installer, install it and then remove it
ADD https://astral.sh/uv/install.sh /install.sh
RUN chmod -R 755 /install.sh && /install.sh && rm /install.sh

# Set up the UV environment path correctly
ENV PATH="/root/.local/bin:${PATH}"

WORKDIR /BioAutoML-FAST

COPY . .

ENV PATH="/BioAutoML-FAST/.venv/bin:${PATH}"

EXPOSE 8501

CMD ["bash", "-c", "uv sync && bash"]