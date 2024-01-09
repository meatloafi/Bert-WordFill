FROM python:3.8

WORKDIR /app

COPY . /app
COPY static /app/static

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 5000

ENV NAME World

CMD ["python", "backend.py"]

