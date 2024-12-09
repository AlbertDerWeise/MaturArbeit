import os
import wave
import librosa
import numpy as np

def split_wav(input_file, output_dir, chunksize: int):
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with wave.open(input_file, 'rb') as wav_file:
        frame_rate = wav_file.getframerate()
        frame_count = wav_file.getnframes()
        duration = frame_count / float(frame_rate)

        # Calculate number of 2-second chunks
        num_chunks = int(duration // chunksize)

        # Read and write 2-second chunks
        for i in range(num_chunks):
            output_file = os.path.join(output_dir, f"chunk_{i + 1}.wav")
            start_frame = int(i * frame_rate * chunksize)
            end_frame = int((i + 1) * frame_rate * chunksize)
            wav_file.setpos(start_frame)
            frames = wav_file.readframes(end_frame - start_frame)

            with wave.open(output_file, 'wb') as out_wav_file:
                out_wav_file.setparams(wav_file.getparams())
                out_wav_file.writeframes(frames)




split_wav(os.path.join('../assets', 'models', 'ff.wav'), os.path.join('../assets','models', 'dude4'), chunksize=2)