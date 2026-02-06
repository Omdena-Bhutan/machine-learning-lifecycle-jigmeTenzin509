FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY ../models ./models
COPY ../src ./src
COPY api.py .

EXPOSE 5000

CMD ["python", "api.py"]
