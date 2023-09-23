from flask import Flask, jsonify, request, render_template, make_response
from flask_restful import Resource, Api
from pybase64 import b64decode
from gradio_client import Client
import openai, requests
from tenacity import retry, wait_random_exponential, stop_after_attempt
import my_secrets as secrets
import json
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

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

class STT():
    def __init__(self):
        self.API_URL = "https://sanchit-gandhi-whisper-jax.hf.space/"
        self.client = Client(self.API_URL)

    def transcribe(self, audio_path):
        text, runtime = self.client.predict(audio_path, "transcribe",False,api_name="/predict_1")
        return text, runtime

class GPTreq(Resource):
    """
    A dummy resource. Useful to test in-browser whether cloud deployment was successful.
    """
    def __init__(self):
        self.gpt = ChatCompletion()

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


class AudioEndpoint(Resource):
    def post(self):

        if 'audioFile' not in request.form:
            val = 'not received'
            out = {"audio":val}

            return out, 404

        val = "received"
        data = request.form['audioFile']
        
        decoded = b64decode(data)
        filename = "audio.caf"

        with open(filename, 'wb') as audio_file:
            audio_file.write(decoded)
        
        transciber = STT()
        text, runtime = transciber.transcribe(filename)
        print(text)

        out = {"text":text, "runtime":runtime}
        return out, 200

class TextAnalysis(Resource):
    def __init__(self):
        self.user_text = None
        self.wizard_text = None
    
    def post(self):
        data = request.json
        print(data)
        self.user_text = data["user_text"]
        self.wizard_text = data["wizard_text"]

        out = {"similarity":self.get_similarity()}
        
        return out, 200
    
    def get_similarity(self):
        tokens1 = nltk.word_tokenize(self.user_text)
        tokens2 = nltk.word_tokenize(self.wizard_text)
        print(tokens1)
        print(tokens2)

        vectorizer = CountVectorizer(input='content', stop_words=None, analyzer=lambda x:x, lowercase=False).fit_transform([tokens1, tokens2])

        # Calculate cosine similarity
        cosine_sim = cosine_similarity(vectorizer)
        
        return int(cosine_sim[0][1]*100)


api.add_resource(GPTreq, '/gptreq')
api.add_resource(AudioEndpoint, "/audio")
api.add_resource(TextAnalysis, "/text-analysis")

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8080, debug=True)