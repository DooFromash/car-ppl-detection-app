FROM python:3.7.16

WORKDIR /app

COPY . /app


RUN pip install ultralytics
RUN pip install opencv-python-headless
RUN pip install -r requirements.txt




EXPOSE 5000

ENV FLASK_APP=app.py

CMD ["python", "app.py"]