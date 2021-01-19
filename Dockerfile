FROM python:3.7-slim
WORKDIR /project
ADD . /project
RUN pip3 --no-cache-dir install -r requirements.txt 
#The cache is usually useless in a Docker image, and you can definitely shrink the image size by disabling the cache.
EXPOSE 5000
CMD ["python3","flask_api.py"]

