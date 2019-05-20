import requests
import os

midis = [
    "https://bitmidi.com/uploads/73866.mid",
    "https://bitmidi.com/uploads/73865.mid",
    "https://bitmidi.com/uploads/73867.mid"
]

os.makedirs("midi_songs", exist_ok=True)
if os.path.exists("data/notes"):
    os.remove("data/notes")

i = 0
for url in midis:
    try:
        r = requests.get(url, allow_redirects=True)
        open("midi_songs/" + url.split("/")[-1], "wb").write(r.content)
        i += 1
    except Exception:
        print("Failed to download %s" % url)

print("Downloaded %i midi files, you can now run `python gru-composer.py`" % i)
