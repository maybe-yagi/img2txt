FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

WORKDIR /usr/src/app
ENV FLASK_APP=app.py
ENV FLASK_ENV=development

COPY requirements.txt ./
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

COPY . .

CMD ["python", "app.py"]
