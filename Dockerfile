FROM python
WORKDIR /app
COPY . /app
CMD ["python3", "app.py"]

# docker build -t housepriceprediction .
# docker run --name housepriceprediction housepriceprediction