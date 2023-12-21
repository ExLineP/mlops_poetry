import dvc.api

url = "https://drive.google.com/drive/folders/1g_uUEWm2VRG0aOM42GtZp7G-m8f492ur"

modelpkl = dvc.api.read(
    'files/hw_task/twitter.csv',
    repo=url,
    mode='rb'
)

print(modelpkl)