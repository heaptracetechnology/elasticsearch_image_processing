Introduction:
    This is flask api to find the visibility score of image.
    It classifies the image into blurry and non-blurry according to their visibility score.
    We used python-opencv library along with other several modules.
    For the full documentation of code.Visit the reference website 
    https://www.pyimagesearch.com/2015/09/07/blur-detection-with-opencv/


Requirements:
    Opencv-python
    Numpy
    Flask
    PIL
    Elasticsearch

installation:
    pip3 install opencv-python numpy flask pillow elasticsearch
   

Configuration and Execution:

    1.start the elasticsearch service by following command:
        systemctl start elasticsearch
    
    2.Run the flask code.
        python3 flask_api.py
    
    3.Hit the endpoint    http://localhost:5000/

    4.upload the images and check the visibility score.

    5.images are also stored in elasticsearch. to check this,hit the following endpoint:
        http://localhost:9200/data/_search?size=1000&from=0

   
    Execution from the Docker file:

    1.pull docker image from following link.
        docker pull swapnil3024/flask-opencv:latest


    2.execute the following command
        docker run --network=host opencv-flask

    
