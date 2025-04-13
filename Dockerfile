# Use Debian as the base image
FROM python:3.11-slim-bookworm

# Install necessary system packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    build-essential \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Download and install uv to /usr/local/bin (accessible to all users)
ADD https://astral.sh/uv/install.sh /install.sh
RUN chmod +x /install.sh && \
    UV_INSTALL_DIR=/usr/local/bin /install.sh && \
    rm /install.sh

# Create a non-root user and home directory
RUN useradd -m user

# Set working directory under the user's home
WORKDIR /home/user/BioAutoML-FAST

# Copy the application code and fix ownership
COPY . /home/user/BioAutoML-FAST
RUN chown -R user:user /home/user/BioAutoML-FAST

# Activate virtual environment path
ENV PATH="/home/user/BioAutoML-FAST/.venv/bin:${PATH}"

# Switch to the non-root user
USER user

# Expose the port
EXPOSE 8501

# Default command
CMD ["bash", "-c", "uv sync && cd App && streamlit run app.py"]
