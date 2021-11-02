FROM agrigorev/zoomcamp-model:3.8.12-slim

WORKDIR /app
COPY ["requirments.txt", "./"]
RUN pip install -r requirements.txt

COPY ["predict.py", "."]

EXPOSE 8088

ENTRYPOINT ["waitress-serve", "--port=8088", "predict:app"]
