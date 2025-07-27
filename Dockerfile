FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy all project files into /app in the container
COPY . /app

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

#5. Expose port 
EXPOSE 8200 
#6. Command to run the server 
CMD ["uvicorn", "iris_fastapi:app", "--host", "0.0.0.0", "--port", "8200"] 
