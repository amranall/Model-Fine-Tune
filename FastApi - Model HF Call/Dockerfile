# Use an official Python image as a base
FROM python:3.12-slim

# Create a new user
RUN useradd -m user

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

RUN  mkdir /.cache

# Change ownership of the application directory to the new user
RUN chown -R user:user /.cache

RUN chown -R user:user /app


# Copy the rest of the application code
COPY . .

# Expose port 8000 for the FastAPI app
EXPOSE 7860

# Switch to the new user
USER user

# Run the FastAPI app with uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
