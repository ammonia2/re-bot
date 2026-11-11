import json
import queue
import threading
import urllib.request
import zipfile
from pathlib import Path

from voice.commands import grammarPhrases

modelsFolder = Path(__file__).resolve().parents[2] / "models"
modelUrl = "https://alphacephei.com/vosk/models/{name}.zip"
sampleRate = 16000


def ensureModel(name):
    folder = modelsFolder / name
    if not folder.is_dir():
        print(f"voice: downloading the speech model {name} (about 40 MB, first run only)")
        archive = modelsFolder / f"{name}.zip"
        urllib.request.urlretrieve(modelUrl.format(name=name), archive)
        with zipfile.ZipFile(archive) as bundle:
            bundle.extractall(modelsFolder)
        archive.unlink()
    return folder


def makeRecognizer(modelName):
    import vosk

    vosk.SetLogLevel(-1)
    model = vosk.Model(str(ensureModel(modelName)))
    return vosk.KaldiRecognizer(model, sampleRate, json.dumps(grammarPhrases()))


def heardText(recognizer, audio):
    # feed raw 16 kHz mono int16 audio; returns a finished sentence or None
    if not recognizer.AcceptWaveform(audio):
        return None
    text = json.loads(recognizer.Result()).get("text", "").replace("[unk]", "").strip()
    return text or None


class speechListener:
    # listens to the microphone on a background thread, fully offline, and queues what it hears
    def __init__(self, config):
        self.modelName = config.get("voiceModel", "vosk-model-small-en-us-0.15")
        self.heard = queue.Queue()
        self.thread = threading.Thread(target=self._listen, daemon=True)

    def start(self):
        self.thread.start()

    def poll(self):
        sentences = []
        while not self.heard.empty():
            sentences.append(self.heard.get_nowait())
        return sentences

    def _listen(self):
        try:
            import sounddevice

            recognizer = makeRecognizer(self.modelName)
            audio = queue.Queue()
            with sounddevice.RawInputStream(samplerate=sampleRate, blocksize=4000, dtype="int16", channels=1,
                                            callback=lambda data, frames, time, status: audio.put(bytes(data))):
                print("voice: listening (say 'follow me', 'follow red', 'next', 'stop', 'go')")
                while True:
                    text = heardText(recognizer, audio.get())
                    if text:
                        self.heard.put(text)
        except Exception as error:
            print(f"voice: microphone commands are off ({error}); the keyboard still works")
