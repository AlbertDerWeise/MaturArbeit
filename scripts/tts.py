import os.path
from TTS.api import TTS
import wave
import struct


#install espeak unless you want to experience death itself!
PathError = type("MissingDirectory", (Exception,), {"__init__": lambda self, path: Exception.__init__(self, f"Specified directory could not be found: {path}")})

#enable gpu if and only if cuda is available
def convert(text: str, save_path: str, voice_model: str = "tts_models/en/jenny/jenny", character: str = '',gpu_enable=False, exist: bool = True) -> None:
    if exist == True:
        print(voice_model, character, save_path)
        tts_engine = TTS(voice_model, gpu=gpu_enable)
        'wav_data = tts_engine.tts(text)'
        save_path1 = list(save_path.removesuffix('.wav'))[::-1]
        for index, element in enumerate(save_path1):
            if element != '/':
                save_path1[index] = ''
            else:
                break

        save_path1 = ''.join(save_path1[::-1])

        if os.path.exists(save_path1):
            try:
                tts_engine.tts_to_file(
                    text=text,
                    file_path=save_path,
                    speaker_wav=[os.path.join(os.getcwd(), '../assets', 'models', character, entry) for entry in os.listdir(os.path.join(os.getcwd(),'../assets','models', character))],
                    language="en",
                    split_sentences=True
                )
            except ValueError or AttributeError as v:
                print(v, ' using predetermined model language as well as speaker instead')
                tts_engine.tts_to_file(
                    text=text,
                    file_path=save_path,
                    # speaker="Ana Florence",
                    # language="en",
                    split_sentences=True
                )
        else:
            raise PathError(save_path1)

        ''''with open(file_path, "wb") as f:
            f.write(wav_data)'''
        print(f"Speech saved as {save_path}")
    else:
        with wave.open(save_path, "w") as wav_file:
            wav_file.setparams((1, 2, 44100, 0, 'NONE', 'not compressed'))
            for _ in range(9 * wav_file.getframerate()):
                wav_file.writeframes(struct.pack('<h', 0))
        print(f'empty speech saved as {save_path}')

