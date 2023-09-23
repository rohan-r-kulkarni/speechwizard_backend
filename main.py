from flask import Flask, jsonify, request, render_template, make_response
from flask_restful import reqparse, Resource, Api
from pybase64 import b64decode
import os
from gradio_client import Client
import openai, requests
from tenacity import retry, wait_random_exponential, stop_after_attempt
import my_secrets as secrets
import json

API_URL = "https://sanchit-gandhi-whisper-jax.hf.space/"

app = Flask(__name__)
api = Api(app)


openai.api_key = secrets.gpt_api_key
GPT_MODEL = "gpt-3.5-turbo-0613"

def pprint(x):
    return json.dumps(x, indent=2)

class ChatCompletion():
    def __init__(self):

       self.chat_start = [
        {"role": "system", "content": "You are speech training app that is to suggest passages to be read aloud, that can help struggling students with pronunciation and ease of speech (i.e. ESL students, students with speech disabilities)"}
        ]

    @retry(wait=wait_random_exponential(min=1, max=40), stop=stop_after_attempt(3))
    def chat_completion_request(self, message, model=GPT_MODEL):
        headers = {
            "Content-Type": "application/json",
            "Authorization": "Bearer " + openai.api_key,
        }
        chat_gpt = self.chat_start.copy()
        chat_gpt.append({"role":"user", "content":message})
        json_data = {"model": model, "messages": chat_gpt}
        try:
            response = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers=headers,
                json=json_data,
            )
            return response
        except Exception as e:
            print("Unable to generate ChatCompletion response")
            print(f"Exception: {e}")
            return e

class GPTreq(Resource):
    """
    A dummy resource. Useful to test in-browser whether cloud deployment was successful.
    """
    def __init__(self):
        self.gpt = ChatCompletion()

    # def get(self):
    #     headers = {"content-type":"text"}
    #     default_message = 
    #     message = self.gpt.chat_completion_request("")
    #     resp = make_response("hello", 200, headers)
    #     return resp

    def post(self):
        req = request.json

        if len(req) == 0: #blank request
            pass
        else:
            out = {"res":"TODO TODO TODO"}

        default_message = "Make ONLY 5 short sentences (7 words or less!) that do not use very complicated english to tell a story. DO NOT SACRIFICE GRAMMAR FOR BEING SUCCINT. Write them as a paragraph." 
        res = self.gpt.chat_completion_request(default_message).text
        data = json.loads(res)["choices"][0]["message"]["content"]
        data_list = [(s.strip() + ".") for s in data.split(".")[:-1]]
        out = {"res":data_list}

        return out, 200

api.add_resource(GPTreq, '/gptreq')

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8080, debug=True)