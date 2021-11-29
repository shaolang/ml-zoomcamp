FROM agrigorev/zoomcamp-cats-dogs-lambda:v2

COPY requirements.txt .
COPY homework-09.py .

RUN pip install -r requirements.txt

ENTRYPOINT ["python", "homework-09.py"]
