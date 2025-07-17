# Use the official Python 3.11-slim image as the base image
FROM python:3.11-slim
# This provides a lightweight, modern Python environment that meets our minimum requirements (Python 3.10+)

# Set the working directory inside the container to /app
WORKDIR /app
# All subsequent commands will execute within this directory

# Create the workspace directory for command execution
RUN mkdir -p /workspace

# Copy all files from the current directory (terminal_server code) into /app in the container
COPY . /app
# This transfers your complete server implementation to the container

# Install required Python packages from requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
# The --no-cache-dir flag reduces image size by preventing pip from caching installation files
# Ensure your requirements.txt includes all dependencies required by your terminal server

# Expose port 5000 for incoming connections
EXPOSE 5000
# Modify this port number if your server is configured to listen on a different port

# Define the command to run your terminal server
CMD ["python", "terminal_server.py"]
# This command launches your terminal server when the container starts