import requests
import config
import time

# From local
BASE_URL = "http://127.0.0.1:8080"

# From Docker Mac
# BASE_URL = "http://0.0.0.0:5000"

# From Docker Windows
# BASE_URL = "http://127.0.0.1:5000"


# Retrain ML
data = open(config.DATA_LOCATION, encoding='latin-1').read()
response = requests.post("{}/retrain_ml".format(BASE_URL), json=data)
print(response.json())

# Predict ML
data = {'text': ['a kamerák leálltak!', 'ki kellene szállítani ezt a csomagot', 'Sziasztok! Mecséri András (MecseriA) nevû kollégát adjátok kérlek hozzá a WG_TUD_KARBANTARTO munkacsoporthoz. Köszi: Dóri']}
response = requests.post("{}/predict_ml".format(BASE_URL), json=data)
print(response.json())

# # Retrain DL
# data = open(config.DATA_LOCATION, encoding='latin-1').read()
# response = requests.post("{}/retrain_dl".format(BASE_URL), json=data)
# print(response.json())

# # Predict DL
# data = {'text': ['a kamerák leálltak!', 'ki kellene szállítani ezt a csomagot', 'Sziasztok! Mecséri András (MecseriA) nevû kollégát adjátok kérlek hozzá a WG_TUD_KARBANTARTO munkacsoporthoz. Köszi: Dóri']}
# response = requests.post("{}/predict_dl".format(BASE_URL), json=data)
# print(response.json())
#
# for i in range(100):
#     data = {'text': ['a kamerák leálltak!', 'ki kellene szállítani ezt a csomagot', 'a felhasználói fiók lejárt']}
#     start = time.time()
#     response = requests.post("{}/predict_ml".format(BASE_URL), json=data)
#     response = requests.post("{}/predict_dl".format(BASE_URL), json=data)
#     end = time.time()
#     print("Done!\nPrediction time (secs): {:.3f}".format(end - start))
#     print(response.json())