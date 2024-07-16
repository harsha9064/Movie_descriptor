import ffmpeg
import os
from gtts import gTTS


def convert_caption_to_audio(caption, output_path='audio.mp3'):
    tts = gTTS(text=caption, lang='en')
    tts.save(output_path)
    return output_path


def combine_audio_video(video_path, audio_path, output_path='static/final_video.mp4'):
    input_video = ffmpeg.input(video_path)
    input_audio = ffmpeg.input(audio_path)
    ffmpeg.output(input_video, input_audio, output_path, vcodec='copy', acodec='aac', strict='experimental').run(
        overwrite_output=True)
    return output_path
