from flask import Flask, jsonify, request, render_template, make_response
from flask_restful import Resource, Api
from pybase64 import b64decode,b64encode
from gradio_client import Client
import openai, requests
from tenacity import retry, wait_random_exponential, stop_after_attempt
import my_secrets as secrets
import json
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import numpy as np
import tensorflow as tf
# from google.cloud import texttospeech


API_URL = "https://sanchit-gandhi-whisper-jax.hf.space/"

app = Flask(__name__)
api = Api(app)


openai.api_key = secrets.gpt_api_key
GPT_MODEL = "gpt-3.5-turbo-0613"

def pprint(x):
    return json.dumps(x, indent=2)

global mistakes
mistakes = {1:{}, 2:{}}

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
    def __init__(self):
        self.gpt = ChatCompletion()

    def post(self):
        req = request.json
        
        default_message = "Make EXACTLY 5 short sentences (between 3-7 words) that do not use very complicated english to tell a story. DO NOT SACRIFICE GRAMMAR FOR BEING SUCCINT. Write them as a paragraph." 

        if len(req["include"]) == 0: #blank request, coming from Home
            print("Launched from Home, no suggestions were provided.")
            mistakes = {1:{}, 2:{}}
            send_message = default_message
        else:
            print("Suggestions were provided:")
            print(req["include"])
            # EXPLOITATION, EXPLORATION
            epsilon = 0.3
            val = np.random.choice([True,False],replace=True, p=[epsilon, 1-epsilon])
            if val:
                send_message = default_message + " Try to include some of these words and phrases: " + " ".join(req["include"])
            else:
                send_message = default_message

        res = self.gpt.chat_completion_request(send_message).text
        data = json.loads(res)["choices"][0]["message"]["content"]
        data_list = [(s.strip() + ".") for s in data.split(".")[:-1]]
        out = {"res":data_list}

        return out, 200

class AudioEndpoint(Resource):
    def get(self):
        """
        Test method
        """
        return {"res":"test successful"}, 200

    def post(self):

        if 'audioFile' not in request.form:
            val = 'not received'
            out = {"audio":val}

            return out, 404

        val = "received"
        data = request.form['audioFile']
        
        decoded = b64decode(data)
        filename = "/tmp/audio.caf"

        with open(filename, 'wb') as audio_file:
            audio_file.write(decoded)
        
        transciber = STT()
        text, runtime = transciber.transcribe(filename)

        out = {"text":text, "runtime":runtime}
        return out, 200

class TextAnalysis(Resource):
    def __init__(self):
        self.user_text = None
        self.wizard_text = None
    
    # def get(self):
    #     """
    #     Return TTS
    #     """

    #     # Instantiates a client
    #     client = texttospeech.TextToSpeechClient()
    #     synthesis_input = texttospeech.SynthesisInput(text="hello to this beautiful, beautiful world!")
    #     voice = texttospeech.VoiceSelectionParams(
    #         language_code="en-US", ssml_gender=texttospeech.SsmlVoiceGender.NEUTRAL
    #     )
    #     audio_config = texttospeech.AudioConfig(
    #         audio_encoding=texttospeech.AudioEncoding.MP3
    #     )
    #     response = client.synthesize_speech(
    #         input=synthesis_input, voice=voice, audio_config=audio_config
    #     )
    #     # The response's audio_content is binary.
    #     filepath = "/tmp/output.mp3"
    #     with open(filepath, "wb") as out:
    #         # Write the response to the output file.
    #         out.write(response.audio_content)

    #     with open(filepath, 'rb') as audio_file:
    #         audio_data = audio_file.read()
    #         base64_string = b64encode(audio_data).decode('utf-8')

    #     return {"bdata":base64_string}, 200
        

    def post(self):
        data = request.json
        self.user_text = data["user_text"]
        self.wizard_text = data["wizard_text"]

        self.update_mistakes()
        gram_suggest = self.k2_max_mistakes(2)

        out = {"similarity":self.get_similarity(), "suggestions":gram_suggest}
        
        return out, 200
    
    def clear_mistakes(self):
        mistakes = {1:{}, 2:{}}

    def clean_text(self, text):
        text = text.replace(".", " ")
        text = text.replace(",", "")
        text = text.replace("!", "")
        text = text.replace("?", "")
        text = text.replace("\'", "")
        text = text.lower()

        return text

    def get_ngrams(self, text, n):
        text = self.clean_text(text)
        words = [word.strip() for word in text.split(" ")]
        return tf.strings.ngrams(words, n)

    def update_mistakes(self):
        for nlen in [1,2]:
            mistake = mistakes[nlen]

            good_gram = self.get_ngrams(self.wizard_text, nlen).numpy()
            bad_gram = self.get_ngrams(self.user_text, nlen).numpy()

            for i in range(len(good_gram)):
                gram_key = good_gram[i]

                if i == len(bad_gram):
                    break

                if gram_key == bad_gram[i]:
                    mistake[gram_key] = mistake.get(gram_key, 0) - 1 #if there was a mistake, you got better
                elif (i > 0 and gram_key == bad_gram[i-1]) or (i < len(bad_gram)-1 and gram_key == bad_gram[i+1]):
                    continue #close calls are not penalized
                else: #mistake found
                    mistake[gram_key] = mistake.get(gram_key, 0) + 1
    
    def k2_max_mistakes(self, k):
        k2_out = []
        for nlen in [1,2]:
            mistake = mistakes[nlen]
            item_list = list(mistake.items())
            item_list.sort(key=lambda kv: kv[1], reverse=True)
            k2_out.extend([tup[0].decode('ascii') for tup in item_list[:k]])
        return k2_out


    def get_similarity(self):
        tokens1 = nltk.word_tokenize(self.clean_text(self.user_text))
        tokens2 = nltk.word_tokenize(self.clean_text(self.wizard_text))

        vectorizer = CountVectorizer(input='content', stop_words=None, analyzer=lambda x:x, lowercase=False).fit_transform([tokens1, tokens2])

        # Calculate cosine similarity
        cosine_sim = cosine_similarity(vectorizer)

        if int(cosine_sim[0][1]*100) > 96:
            return 100
        else:
            return int(cosine_sim[0][1]*100)


api.add_resource(GPTreq, '/gptreq')
api.add_resource(AudioEndpoint, "/audio")
api.add_resource(TextAnalysis, "/text-analysis")

if __name__ == "__main__":
    # context = ndb.get_context()
    # context.set_cache_policy(func)
    # context.set_memcache_policy(func)

    app.run(host='0.0.0.0', port=8080, debug=True)