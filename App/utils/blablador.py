import requests
import json


class Models():
   
    def __init__(self, api_key):
        self.api_key = api_key
        self.headers = {'accept': 'application/json', 'Authorization': f'Bearer {api_key}'}
   
    url = "https://helmholtz-blablador.fz-juelich.de:8000/v1/models"
     
    def get_model_data(self):
        response = requests.get(url = self.url, headers = self.headers)
        response = json.loads(response.text)
        return(response["data"])

    def get_model_ids(self):
        response = requests.get(url = self.url, headers = self.headers)
        response = json.loads(response.text)

        # TODO write error messages for 400, 401, etc respones
        # like with response.ok , response.status, etc... 

        ids = []
        for model in response["data"]:
            ids.append(model["id"])

        return(ids)

class ChatCompletions():

    def __init__(self, api_key, model, temperature = 0.7, choices =  1, max_tokens = 500, user = 'default'):
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.choices = choices
        self.max_tokens = max_tokens
        self.user = user
        self.headers = {'accept': 'application/json', 'Authorization': f'Bearer {api_key}', 'Content-Type': 'application/json'}

   
    url = "https://helmholtz-blablador.fz-juelich.de:8000/v1/chat/completions"
    
    # don't know what these are, using default values from https://helmholtz-blablador.fz-juelich.de:8000/docs#/
    top_p =  1 # has something to do with temperature...
    presence_penalty = 0
    frequency_penalty = 0

    def get_completion(self, messages):
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "n": self.choices,
            "max_tokens": self.max_tokens,
            "stop": [
                "string"
            ],
            "stream": "false",
            "presence_penalty": self.presence_penalty,
            "frequency_penalty": self.frequency_penalty,
            "user": self.user
        }
        payload = json.dumps(payload)
        
        response = requests.post(url = self.url, headers = self.headers, data=payload)
        # TODO write error messages for 400, 401, etc respones
        # like with response.ok , response.status, etc... 
        return(response.text)

class Completions():

    def __init__(self, api_key, model,temperature = 0.7, choices = 1, max_tokens =  50, user = "default"):
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.choices = choices
        self.max_tokens = max_tokens
        self.user = user

        self.headers = {'accept': 'application/json', 'Authorization': f'Bearer {api_key}', 'Content-Type': 'application/json'}

   
    url = "https://helmholtz-blablador.fz-juelich.de:8000/v1/completions"
    
    # don't know what these are, using default values from https://helmholtz-blablador.fz-juelich.de:8000/docs#/

    suffix = "string"
    logprobs = 0
    echo = "false"
    top_p =  1 # has something to do with temperature...
    presence_penalty = 0
    frequency_penalty = 0

    def get_completion(self, prompt):
        payload = {
            "model": self.model,
            "prompt": prompt,
            "suffix": self.suffix,
            "temperature": self.temperature,
            "n": self.choices,
            "max_tokens": self.max_tokens,
            "stop": [
                "string"
            ],
            "stream": "false",
            "top_p": self.top_p,
            "logprobs":self.logprobs,
            "echo":self.echo,
            "presence_penalty": self.presence_penalty,
            "frequency_penalty": self.frequency_penalty,
            "user": self.user
        }

        payload = json.dumps(payload)
        
        response = requests.post(url = self.url, headers = self.headers, data=payload)
        # TODO write error messages for 400, 401, etc respones
        # like with response.ok , response.status, etc... 
        return(response.text)
    
class TokenCount():
     
    def __init__(self, model, max_tokens = 0):
        self.model = model
        self.max_tokens = max_tokens #this does nothing
        self.headers = {'accept': 'application/json', 'Content-Type': 'application/json'}

    
    url = "https://helmholtz-blablador.fz-juelich.de:8000/api/v1/token_check"


    def count(self, prompts):
        try:
            iterator = iter(prompts)
        except TypeError:
            prompt_list = [ 
                    {
                        "model": "zephyr-7b-beta",
                        "prompt": prompts,
                        "max_tokens":self.max_tokens
                    } 
                ]
        else:
            prompt_list = []
            for prompt in prompts:
                prompt_list.append(
                    {
                    "model": "zephyr-7b-beta",
                    "prompt": prompt,
                    "max_tokens": self.max_tokens
                    }   
                )

        payload = {
                "prompts": prompt_list
            }
        
        payload = json.dumps(payload)
        
        response = requests.post(url = self.url, headers = self.headers, data=payload)
        # TODO write error messages for 400, 401, etc respones
        # like with response.ok , response.status, etc... 
        return(response.text)

#embeddings_url = "https://helmholtz-blablador.fz-juelich.de:8000/v1/embeddings"
# model_embeddings_url = "https://helmholtz-blablador.fz-juelich.de:8000/v1/engines/{model_name}/embeddings"
