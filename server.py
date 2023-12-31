from copy import deepcopy
import io
import json
import time
from flask import Flask, jsonify, request, render_template
import numpy as np
import os 
from keras_facenet import FaceNet
from sklearn.cluster import DBSCAN
import subprocess
from typing import List
from flask_cors import CORS
import base64
import logging
import rawpy
from PIL import Image
logging.getLogger("tensorflow").setLevel(logging.ERROR)
import cv2
app = Flask(__name__)
CORS(app)
embedder = FaceNet()
fnames_table = {}
saved_embeddings = []

def get_image_paths(folder_name: str):

    if folder_name in fnames_table:
        return fnames_table[folder_name]
    
    image_paths = []
    for root, dirs, files in os.walk(folder_name):
        for file in files:
            if file.lower().endswith(("jpg", "jpeg", "png", "gif", 
                                        "bmp", "tiff", "tif","webp","cr2"
                                    )):
                image_paths.append(os.path.join(root, file))
    fnames_table[folder_name] = image_paths
    return image_paths

def find_faces(image_path:str):
    try :
        print(image_path)
        if 'cr2' in image_path.lower():
            with rawpy.imread(image_path) as raw:
                image = raw.postprocess()
        else:
            image = np.asarray(Image.open(image_path))
        if image is None:
            app.logger.error(f"Erreur lors de la lecture de l'image {image_path}")
            return []
        print(image.shape)
        if image.shape[0] > 800 or image.shape[1] > 800:
            image = cv2.resize(image, (int(image.shape[1]*(700/image.shape[0])) , 
                                       int(image.shape[0]* (700/image.shape[0]))))

        detections = embedder.extract(image, threshold=0.8)
        print("Detection done")
        for detection in detections:
            detection["image_path"] = image_path
        del image
        return detections
    except Exception as e:
        app.logger.error(f"Erreur lors de la recherche de visages : {e}")
        return []

def cluster(detections: List[dict], eps: float, min_samples: int):
    try:
        print("Begin clustering")
        
        for d in detections:
            d["embedding"] = [float(v) for v in d["embedding"]]

        with open("save_emb.json","w") as f:
            json.dump(detections, f)

        embeddings = np.array([d["embedding"] for d in detections], dtype=np.float32)
        print("Nb embedding", len(embeddings))
        print("Size of embedding", embeddings[0].shape)
        pred = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(embeddings)
        print("Ending clustering")
        return pred, [d["image_path"] for d in detections]
    
    except Exception as e:
        app.logger.error(f"Erreur lors de l'exécution de cluster() : {e}")
        return [], []

    

def add_exif_data(image_paths: List[str], face_name: str):
    try:
        chunk_length = 150
        for i in range(0, len(image_paths), chunk_length):
            subprocess.call(["exiftool", "-overwrite_original", "-Subject+="+face_name] + image_paths[i:i+chunk_length])
            app.logger.info(f"Image {i} sur {len(image_paths)} traitée")
    except subprocess.SubprocessError as e:
        app.logger.error(f"Erreur lors de l'exécution de ExifTool : {e}")
        return False
    return True

def convert_to_base64(image_path: str, face_box: List[int]):
    # Load the image differently based on file type
    try:
        if 'cr2' in image_path.lower():
            app.logger.info(f"Image {image_path} is a raw file")
            with rawpy.imread(image_path) as raw:
                image = raw.postprocess()
                image = Image.fromarray(image)
        else:
            app.logger.info(f"Image {image_path} is a jpg file")
            image = Image.open(image_path)

        # Check if image is loaded
        if image is None:
            app.logger.error(f"Erreur lors de la lecture de l'image (convert b64) {image_path}")
            return "", False

        # Resize if needed using Pillow
        if image.size[0] > 800 or image.size[1] > 800:
            scaling_factor = 700 / min(image.size[0], image.size[1])
            new_size = (int(image.size[0] * scaling_factor), int(image.size[1] * scaling_factor))
            image = image.resize(new_size, Image.Resampling.LANCZOS)

        x, y, w, h = face_box
        cropped_image = image.crop((x, y, x+w, y+h))
        # Convert to base64
        buffered = io.BytesIO()
        cropped_image.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')

        return img_str, True 
    
    except Exception as e:
        app.logger.error(f"Erreur lors de convert to b64 : {e}")
        return "", False

     

@app.post("/get_number_of_images")
def get_number_of_images():
    try:
        data = request.get_json()
        folder_name = data["fname"]
        image_paths = get_image_paths(folder_name)
        return jsonify({"number_of_images": len(image_paths)})
    except Exception as e:
        app.logger.error(f"Erreur lors de la récupération du nombre d'images : {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/find_clusters", methods=["POST"])
def find_clusters():
    try:
        # data = request.get_json()
        # dbscan_eps = data.get("dbscan_eps", 0.90)
        # dbscan_min_samples = data.get("dbscan_min_samples", 5)
        # images_path = get_image_paths(data["fname"])
        # all_detections = []

        # for i, image_path in enumerate(images_path):
        #     detections = find_faces(image_path)
        #     all_detections.extend(detections)
        #     if i % 50 == 0:
        #         app.logger.info(f"Image {i} sur {len(images_path)} traitée")
        # if len(all_detections) == 0:
        #     print("Pas de visage trouvé...")
        #     return jsonify({"error": "Aucun visage trouvé dans le dossier"}), 500
        
        with open("save_emb.json","r") as f:
            all_detections = json.load(f)
        dbscan_eps = 1
        dbscan_min_samples = 20
        predicted_clusters, paths = cluster(all_detections, dbscan_eps, dbscan_min_samples) 
        
        res_cluster = {}
        for i, p in enumerate(predicted_clusters):
            if p != -1:
                if p not in res_cluster:
                    res_cluster[int(p)] = { "faces": [], "paths": []}
                res_cluster[int(p)]["paths"].append(paths[i])
                if len(res_cluster[int(p)]["faces"]) < 3:
                    base64_image, success = convert_to_base64(paths[i], all_detections[i]["box"])
                    if success:
                        res_cluster[int(p)]["faces"].append(base64_image)
        return jsonify(res_cluster)
    
    except Exception as e:
        app.logger.error(f"Erreur lors de la recherche de clusters : {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/change_exif", methods=["POST"])
def change_exif():
    try:
        data = request.get_json()
        for cluster_number, cluster in data.items():
            face_name = cluster["name"] 
            image_paths = cluster["paths"]
            if face_name and not add_exif_data(image_paths, face_name):
                raise ValueError("Erreur lors de la modification des métadonnées Exif")
        return jsonify({"status": "ok"})
    except Exception as e:
        app.logger.error(f"Erreur lors de la modification des métadonnées Exif : {e}")
        return jsonify({"error": str(e)}), 500


@app.post("/get_image_by_face")
def get_image_by_face():
    try :
        exiftool_path = ".\exiftool.exe"
        data = request.get_json()
        folder_name = data["fname"]
        faces_to_find = data["face_name"]
        file_paths = get_image_paths(folder_name)
        #cut file paths in chunck of 100 to avoid exiftool error
        chunk_length = 150
        all_results = ""
        for i in range(0, len(file_paths), chunk_length):
            result = subprocess.run([exiftool_path, "-overwrite_original", "-Subject"] + file_paths[i:i+chunk_length],
                                    capture_output=True, text=True)
            all_results += result.stdout
            app.logger.info(f"Image {i} sur {len(file_paths)} traitée")

        matching_files = []
        current_file = None
        for line in all_results.split("\n"):
            if line.startswith('========'):
                current_file = line.split(' ')[1]
            elif 'Subject' in line and all(face.strip() in line.lower() for face in faces_to_find.lower().split(",")):
                matching_files.append(current_file)
        return jsonify(matching_files)
    
    except Exception as e:
        app.logger.error(f"Erreur lors de la récupération des métadonnées Exif : {e}")
        
        return jsonify({"error": str(e)}), 500

@app.get("/")
def index():
    # cannot use serveur because of security issues : i can't access the file system
    # return render_template("index.html")   
    # return hello_world()
    return "Hello World !"


if __name__ == '__main__':
   app.run(port=5000, debug=True)