FROM python:3.11-slim

WORKDIR /usr/src/app
ENV FLASK_APP=app.py
ENV FLASK_ENV=development

COPY requirements.txt ./
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

CMD ["python", "/usr/src/app/app.py"]
