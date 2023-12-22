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

## 3. Usage

### 3.1. Clustering
The first tab is used to cluster faces in images.
You can write down a folder path (absolute path) and click on the button to start the clustering.
When it's done, you can see all the detected people represented by 3 images of them.
You can then put the real name of the person in the input field or leave it blank if you don't know who it is.
You can then click on the button to save the metadata in the images.

### 3.2. Search
The second tab is used to search for images of a given person.
You can write down a folder path and the name of the person you want to search for and click on the button to start the search.
When it's done, you can see all the images of the person you searched for.
You can also write several names separated by a comma to search for several people at the same time.

