import dvc.api

modelpkl = dvc.api.read(
    'twitter.csv',
    repo='https://drive.google.com/drive/folders/1g_uUEWm2VRG0aOM42GtZp7G-m8f492ur',
    mode='rb'
)

print(modelpkl)