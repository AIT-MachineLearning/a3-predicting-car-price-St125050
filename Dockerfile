# Use the official Python image from the Docker Hub
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt ./

# Install the Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code and model file into the container
COPY app.py ./
COPY Ridge_Logistic_Regression_Model.pkl ./

# Expose the port that Streamlit will run on
EXPOSE 8505

# Command to run the Streamlit app
CMD ["streamlit", "run", "app.py"]
