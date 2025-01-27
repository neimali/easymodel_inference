import os
import shutil
from datetime import datetime
from urllib.parse import urlparse
from flask import Flask, request, jsonify
import requests
from PIL import Image
import imageio
import boto3
from botocore.exceptions import BotoCoreError, ClientError
from config import Config
from trellis.pipelines import TrellisImageTo3DPipeline
from trellis.utils import render_utils, postprocessing_utils

def inference(imagePath, modelName="JeffreyXiang/TRELLIS-image-large"):
    pipeline = TrellisImageTo3DPipeline.from_pretrained(modelName)
    pipeline.cuda()

    # Load an image
    image = Image.open(imagePath)

    # Run the pipeline
    outputs = pipeline.run(
        image,
        seed=1,
        # Optional parameters
        # sparse_structure_sampler_params={
        #     "steps": 12,
        #     "cfg_strength": 7.5,
        # },
        # slat_sampler_params={
        #     "steps": 12,
        #     "cfg_strength": 3,
        # },
    )
    return outputs

app = Flask(__name__)
app.config.from_object(Config)
os.makedirs(Config.TEMP_DIR, exist_ok=True)

@app.route('/inference', methods=['POST'])
def get_image(): #todo add proper path for image
    try:
        # 从请求中获取预签名 URL
        data = request.get_json()
        presigned_url = data.get("presigned_url")
        userData = data.get("userData")
        
        if not presigned_url:
            return jsonify({"error": "No presigned URL provided"}), 400

        # 下载图片到临时目录
        parsed_url = urlparse(presigned_url)
        filename = os.path.basename(parsed_url.path)
        if not filename:
            return jsonify({"error": "Cannot get image file name"}), 500

        response = requests.get(presigned_url)
        if response.status_code != 200:
            return jsonify({"error": "Failed to download image"}), 500

        imagePath = os.path.join(Config.TEMP_DIR, filename)
        with open(imagePath, 'wb') as f:
            f.write(response.content)

        files_download_urls = generate_download_urls(userData, imagePath)

        return jsonify({"message": "Image downloaded successfully", "model_files_presigned_urls": files_download_urls}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500



def generate_download_urls(userData, imagePath, modelName):

    try:
        outputs = inference(imagePath, modelName)
    except Exception as e:
        raise RuntimeError(f"Error during inference: {e}")
    # outputs is a dictionary containing generated 3D assets in different formats:
    # - outputs['gaussian']: a list of 3D Gaussians
    # - outputs['radiance_field']: a list of radiance fields
    # - outputs['mesh']: a list of meshes

    #create local dir to store fies before upload to s3
    user_name = userData['name']
    image_name, _ = os.path.splitext(imagePath)
    files_local_dir = os.path.join(user_name, image_name, "")
    os.makedirs(files_local_dir, exist_ok=True)
    # Render the outputs
    video = render_utils.render_video(outputs['gaussian'][0])['color']
    imageio.mimsave(os.path.join(files_local_dir, "sample_gs.mp4"), video, fps=30)
    video = render_utils.render_video(outputs['radiance_field'][0])['color']
    imageio.mimsave(os.path.join(files_local_dir, "sample_rf.mp4"), video, fps=30)
    video = render_utils.render_video(outputs['mesh'][0])['normal']
    imageio.mimsave(os.path.join(files_local_dir, "sample_mesh.mp4"), video, fps=30)

    # GLB files can be extracted from the outputs
    glb = postprocessing_utils.to_glb(
        outputs['gaussian'][0],
        outputs['mesh'][0],
        # Optional parameters
        simplify=0.95,          # Ratio of triangles to remove in the simplification process
        texture_size=1024,      # Size of the texture used for the GLB
    )
    glb.export(os.path.join(files_local_dir, "sample.glb"))

    # Save Gaussians as PLY files
    outputs['gaussian'][0].save_ply(os.path.join(files_local_dir, "sample.ply"))


    #Upload to AWS S3 and generate presigned-url
    s3 = boto3.client('s3')
    
    files_paths = ['sample_gs.mp4', 'sample_rf.mp4', 'sample_mesh.mp4', 'sample.glb', 'sample.ply']
    #check if files are ready in local
    for file_path in files_paths:
        full_path = os.path.join(files_local_dir, file_path)
        if not os.path.exists(full_path):
            raise RuntimeError(f"File {file_path} does not exist.")
        if os.path.getsize(full_path) == 0:
            raise RuntimeError(f"File {file_path} is empty.")
    #upload
    files_download_urls = {}
    try:
        uploaded_files = []
        for file_path in files_paths:
            s3_key = f"{user_name}/{image_name}/{file_path}"
            s3.upload_file(os.path.join(files_local_dir, file_path), app.config['AWS_STORAGE_BUCKET_NAME'], s3_key)
            file_url = s3.generate_presigned_url('get_object',
                                        Params={'Bucket': app.config['AWS_STORAGE_BUCKET_NAME'], 'Key': s3_key},
                                        ExpiresIn=3600)
            files_download_urls[file_path] = file_url
            uploaded_files.append(s3_key)
    except (BotoCoreError, ClientError) as e:
        for uploaded_file in uploaded_files:
            s3.delete_object(Bucket=app.config['AWS_STORAGE_BUCKET_NAME'], Key=uploaded_file)
        raise RuntimeError(f"Error uploading to S3: {e}")
    
    #clean up local storage, todo: add exception process
    shutil.rmtree(files_local_dir)
    try:
        os.remove(imagePath)
    except FileNotFoundError:
        app.logger.warning(f"File {imagePath} not found during cleanup.")
    except Exception as e:
        app.logger.error(f"Error during cleanup of {imagePath}: {e}")

    return files_download_urls