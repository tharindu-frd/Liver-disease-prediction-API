# Use an official Python runtime as a base image
FROM ubuntu
# Install build-essential package to include gcc compiler
RUN apt-get update && apt-get install -y build-essential

RUN apt install python3-pip -y
RUN pip3 install Flask
# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt



# Run app.py when the container launches
CMD ["python3", "-m","flask","run","--host=0.0.0.0"]