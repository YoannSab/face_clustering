# face_clustering

## 1. Introduction
This app is small web app whith python flask backend.
It is used to find and cluster faces in images so that the names of the people in the images can be stored in the metadata of the images.
It can also be used to find all images of one person (or several) in a given folder based on the metadata.

## 2. Installation

Setup a virtual environment and install the requirements with the following commands:
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```
then use the following command to start the server, it's not meant to be used in production, so it's not secure at all.
```bash
python server.py
```
Then open index.html with you browser file system.
We can't use the server to render the html file because of security policies of the browser.

