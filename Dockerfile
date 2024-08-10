# Use the official Python image from the Docker Hub
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install the dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the container
COPY . .

# Copy the CSV file into the container
COPY Data/application_record.csv /app/application_record.csv
COPY Data/credit_record.csv /app/credit_record.csv
# Expose the port the app runs on
EXPOSE 80

# Run the application
CMD ["python", "app.py"]
