FROM openvino/dev-py:latest

WORKDIR /app

COPY requirements.txt .
RUN python3 -m pip install --no-cache-dir -r requirements.txt

COPY main.py .
COPY ./app ./app

COPY ./best0408_openvino_model ./best0408_openvino_model

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
