# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements.txt first to leverage Docker's cache for dependencies
COPY requirements.txt /app/

# Install dependencies for both FastAPI and Streamlit
RUN pip install --no-cache-dir -r requirements.txt

# Now copy the rest of the application files
COPY . /app

# Expose the ports the apps will run on
EXPOSE 8000 8501

# Command to run both FastAPI and Streamlit using a shell script
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port 8000 & streamlit run app.py --server.port 8501"]


