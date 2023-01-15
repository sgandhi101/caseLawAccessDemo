import requests
import pandas as pd

response = requests.get(
    'https://api.case.law/v1/cases/?page_size=10&search=piracy&ordering=relevance',
    headers={'Authorization': 'Token 519e093c5a6dbbbd322f9a0e931d7b12e4f9d471'}
)

print(response.text)