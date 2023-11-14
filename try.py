from clarifai.client.model import Model
import os
from google.protobuf import json_format

os.environ['CLARIFAI_PAT'] = '95978ef1e65e4e1ab8b268e94a49b1e9'

response = Model(user_id="stability-ai", app_id="Upscale", model_id="stabilityai-upscale").predict_by_filepath("/Users/mansikhamkar/Downloads/dogs/golden-retriever-puppy.jpg", "image")
response = json_format.MessageToDict(response)
print(type(response))
print(response['outputs'][0])