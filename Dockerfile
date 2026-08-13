FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt gunicorn

COPY . .

ENV PORT=8080
EXPOSE ${PORT}

CMD gunicorn --bind 0.0.0.0:${PORT} --timeout 300 --workers 1 railway_server:app
