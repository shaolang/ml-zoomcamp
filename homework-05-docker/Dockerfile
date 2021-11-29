FROM agrigorev/zoomcamp-model:3.8.12-slim

RUN pip install pipenv

WORKDIR /app
COPY ["Pipfile", "Pipfile.lock", "./"]

RUN pipenv install --system --deploy

COPY ["homework5_ws.py", "."]

EXPOSE 8088

ENTRYPOINT ["gunicorn", "--bind 0.0.0.0:8088", "homework5_ws:app"]
