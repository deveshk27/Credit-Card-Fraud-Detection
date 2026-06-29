# 1. Start with a lightweight Linux image containing Python 3.14
FROM python:3.14-slim

# 2. Set the working directory inside the container
WORKDIR /app

# 3. Copy just the requirements file first
COPY requirements.txt .

# 4. Install the Python dependencies inside the container
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copy the rest of your application code
COPY . .

# 6. Expose the port that Uvicorn will run on
EXPOSE 8000

# 7. The terminal command the container runs when it starts up
CMD ["sh", "-c", "uvicorn api:app --host 0.0.0.0 --port ${PORT:-8000}"]