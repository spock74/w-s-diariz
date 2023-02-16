import whisper
import datetime
import subprocess
import gradio as gr
from pathlib import Path
import pandas as pd
import re
import time
import os
import numpy as np
from sklearn.cluster import AgglomerativeClustering

from pytube import YouTube
import torch
import pyannote.audio
from pyannote.audio.pipelines.speaker_verification import PretrainedSpeakerEmbedding
from pyannote.audio import Audio
from pyannote.core import Segment

from gpuinfo import GPUInfo

import wave
import contextlib
from transformers import pipeline
import psutil

whisper_models = ["base", "small", "medium", "large"]
source_languages = {
    "pt": "Portuguese",
    "en": "English",
    # "zh": "Chinese",
    "de": "German",
    "es": "Spanish",
    # "ru": "Russian",
    # "ko": "Korean",
    "fr": "French",
    # "ja": "Japanese",
    # "tr": "Turkish",      
    # "pl": "Polish",
    # "ca": "Catalan",
    # "nl": "Dutch",
    # "ar": "Arabic",
    # "sv": "Swedish",
    # "it": "Italian",
    # "id": "Indonesian",
    # "hi": "Hindi",
    # "fi": "Finnish",
    # "vi": "Vietnamese",
    # "he": "Hebrew",
    # "uk": "Ukrainian",
    # "el": "Greek",
    # "ms": "Malay",
    # "cs": "Czech",
    # "ro": "Romanian",
    # "da": "Danish",
    # "hu": "Hungarian",
    # "ta": "Tamil",
    # "no": "Norwegian",
    # "th": "Thai",
    # "ur": "Urdu",
    # "hr": "Croatian",
    # "bg": "Bulgarian",
    # "lt": "Lithuanian",
    # "la": "Latin",
    # "mi": "Maori",
    # "ml": "Malayalam",
    # "cy": "Welsh",
    # "sk": "Slovak",
    # "te": "Telugu",
    # "fa": "Persian",
    # "lv": "Latvian",
    # "bn": "Bengali",
    # "sr": "Serbian",
    # "az": "Azerbaijani",
    # "sl": "Slovenian",
    # "kn": "Kannada",
    # "et": "Estonian",
    # "mk": "Macedonian",
    # "br": "Breton",
    # "eu": "Basque",
    # "is": "Icelandic",
    # "hy": "Armenian",
    # "ne": "Nepali",
    # "mn": "Mongolian",
    # "bs": "Bosnian",
    # "kk": "Kazakh",
    # "sq": "Albanian",
    # "sw": "Swahili",
    # "gl": "Galician",
    # "mr": "Marathi",
    # "pa": "Punjabi",
    # "si": "Sinhala",
    # "km": "Khmer",
    # "sn": "Shona",
    # "yo": "Yoruba",
    # "so": "Somali",
    # "af": "Afrikaans",
    # "oc": "Occitan",
    # "ka": "Georgian",
    # "be": "Belarusian",
    # "tg": "Tajik",
    # "sd": "Sindhi",
    # "gu": "Gujarati",
    # "am": "Amharic",
    # "yi": "Yiddish",
    # "lo": "Lao",
    # "uz": "Uzbek",
    # "fo": "Faroese",
    # "ht": "Haitian creole",
    # "ps": "Pashto",
    # "tk": "Turkmen",
    # "nn": "Nynorsk",
    # "mt": "Maltese",
    # "sa": "Sanskrit",
    # "lb": "Luxembourgish",
    # "my": "Myanmar",
    # "bo": "Tibetan",
    # "tl": "Tagalog",
    # "mg": "Malagasy",
    # "as": "Assamese",
    # "tt": "Tatar",
    # "haw": "Hawaiian",
    # "ln": "Lingala",
    # "ha": "Hausa",
    # "ba": "Bashkir",
    "jw": "Javanese",
    # "su": "Sundanese",
}

source_language_list = [key[0] for key in source_languages.items()]

MODEL_NAME = "openai/whisper-medium"
lang = "pt"

device = 0 if torch.cuda.is_available() else "cpu"
pipe = pipeline(
    task="automatic-speech-recognition",
    model=MODEL_NAME,
    chunk_length_s=30,
    device=device,
)

pipe.model.config.forced_decoder_ids = pipe.tokenizer.get_decoder_prompt_ids(
    language=lang, task="transcribe")

embedding_model = PretrainedSpeakerEmbedding(
    "speechbrain/spkrec-ecapa-voxceleb",
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))


def transcribe(microphone, file_upload):
    warn_output = ""
    if (microphone is not None) and (file_upload is not None):
        warn_output = (
            "AVISO: Você enviou um arquivo de áudio e usou o microfone."
            "O áudio gravado do microfone será usado e o áudio enviado será descartado.\n"
        )

    elif (microphone is None) and (file_upload is None):
        return "ERROR: Você deve usar o microfone ou enviar um arquivo de áudio"

    file = microphone if microphone is not None else file_upload

    text = pipe(file)["text"]

    return warn_output + text


def _return_yt_html_embed(yt_url):
    video_id = yt_url.split("?v=")[-1]
    HTML_str = (
        '<center> <iframe width="500" height="320" src="https://www.youtube.com/embed/{video_id}"> </iframe>'
        " </center>"
    )
    return HTML_str


def yt_transcribe(yt_url):
    yt = YouTube(yt_url)
    html_embed_str = _return_yt_html_embed(yt_url)
    stream = yt.streams.filter(only_audio=True)[0]
    stream.download(filename="audio.mp3")

    text = pipe("audio.mp3")["text"]

    return html_embed_str, text


def convert_time(secs):
    return datetime.timedelta(seconds=round(secs))


def get_youtube(video_url):
    yt = YouTube(video_url)
    abs_video_path = yt.streams.filter(progressive=True, file_extension='mp4').order_by(
        'resolution').desc().first().download()
    print("Vídeo baixado com sucesso")
    print(abs_video_path)
    return abs_video_path


def speech_to_text(video_file_path, selected_source_lang, whisper_model, num_speakers):
    """
    ## Trancreva vídeo do youtube
    """
    # 1. Usa o modelo de AI 'Whisper' para separar audio em segmentos e gerar transcrições.
    # 2. Geração de 'speaker embeddings' para cada segmento.
    # 3. Aplica clusterização aglomerativa nos 'embeddings' para identificar o falante de cada segmento.
    # Speech Recognition is based on models from OpenAI Whisper https://github.com/openai/whisper Speaker diarization model and pipeline from by https://github.com/pyannote/pyannote-audio

    model = whisper.load_model(whisper_model)
    time_start = time.time()
    if(video_file_path is None):
        raise ValueError("Erro no video input")
    print(video_file_path)

    try:
        # Read and convert youtube video
        _, file_ending = os.path.splitext(f'{video_file_path}')
        print(f'Arquivo de entrada:  {file_ending}')
        audio_file = video_file_path.replace(file_ending, ".wav")
        print("Iniciar conversão para wav")
        os.system(
            f'ffmpeg -i "{video_file_path}" -ar 16000 -ac 1 -c:a pcm_s16le "{audio_file}"')

        # Get duration
        with contextlib.closing(wave.open(audio_file, 'r')) as f:
            frames = f.getnframes()
            rate = f.getframerate()
            duration = frames / float(rate)
        print(
            f"conversão para wav pronta, duração do arquivo de áudio: {duration}")

        # Transcribe audio
        options = dict(language=selected_source_lang, beam_size=5, best_of=5)
        transcribe_options = dict(task="transcribe", **options)
        result = model.transcribe(audio_file, **transcribe_options)
        segments = result["segments"]
        print("::::??? Zehn ???::: starting whisper done with whisper")
    except Exception as e:
        raise RuntimeError("Erro ao converter vídeo para áudio")

    try:
        # Create embedding
        def segment_embedding(segment):
            audio = Audio()
            start = segment["start"]
            # Whisper overshoots the end timestamp in the last segment
            end = min(duration, segment["end"])
            clip = Segment(start, end)
            waveform, sample_rate = audio.crop(audio_file, clip)
            return embedding_model(waveform[None])

        embeddings = np.zeros(shape=(len(segments), 192))
        for i, segment in enumerate(segments):
            embeddings[i] = segment_embedding(segment)
        embeddings = np.nan_to_num(embeddings)
        print(f'::::ZEHN - Embedding shape:::: {embeddings.shape}')

        # Assign speaker label
        clustering = AgglomerativeClustering(num_speakers).fit(embeddings)
        labels = clustering.labels_
        for i in range(len(segments)):
            segments[i]["speaker"] = f"FALANTE {str(labels[i] + 1)}"

        # Make output
        objects = {
            'Start': [],
            'End': [],
            'Speaker': [],
            'Text': []
        }
        text = ''
        for (i, segment) in enumerate(segments):
            if i == 0 or segments[i - 1]["speaker"] != segment["speaker"]:
                objects['Start'].append(str(convert_time(segment["start"])))
                objects['Speaker'].append(segment["speaker"])
                if i != 0:
                    objects['End'].append(
                        str(convert_time(segments[i - 1]["end"])))
                    objects['Text'].append(text)
                    text = ''
            text += segment["text"] + ' '
        objects['End'].append(str(convert_time(segments[i - 1]["end"])))
        objects['Text'].append(text)

        time_end = time.time()
        time_diff = time_end - time_start
        memory = psutil.virtual_memory()
        gpu_utilization, gpu_memory = GPUInfo.gpu_usage()
        gpu_utilization = gpu_utilization[0] if len(gpu_utilization) > 0 else 0
        gpu_memory = gpu_memory[0] if len(gpu_memory) > 0 else 0
        system_info = f"""
        *Memória: {memory.total / (1024 * 1024 * 1024):.2f}GB, usada: {memory.percent}%, disponível: {memory.available / (1024 * 1024 * 1024):.2f}GB.* 
        *Tempo de processamento: {time_diff:.5} segundos.*
        *Utilização de GPU: {gpu_utilization}%, Memória GPU: {gpu_memory}MiB.*
        """

        return pd.DataFrame(objects), system_info

    except Exception as e:
        raise RuntimeError("Erro de inferência ao rodar o modelo local", e)


# ---- Gradio Layout -----
# Inspiration from https://huggingface.co/spaces/RASMUS/Whisper-youtube-crosslingual-subtitles
video_in = gr.Video(label="Arquivo de vídeo", mirror_webcam=False)
youtube_url_in = gr.Textbox(label="URL do vídeo", lines=1, interactive=True)
df_init = pd.DataFrame(columns=['Início', 'Fim', 'Falante', 'Texto'])
memory = psutil.virtual_memory()
selected_source_lang = gr.Dropdown(
    choices=source_language_list, type="value", value="pt", label="Língua", interactive=True)
selected_whisper_model = gr.Dropdown(choices=whisper_models, type="value",
                                     value="medium", label="Selecione o Modelo Whisper", interactive=True)
number_speakers = gr.Number(
    precision=0, value=2, label="Número de falantes", interactive=True)
system_info = gr.Markdown(
    f"*Memoria: {memory.total / (1024 * 1024 * 1024):.2f}GB, usada: {memory.percent}%, disponível: {memory.available / (1024 * 1024 * 1024):.2f}GB*")
transcription_df = gr.DataFrame(value=df_init, label="Dataframe da transcrição", row_count=(
    0, "dynamic"), max_rows=10, wrap=True, overflow_row_behaviour='paginate')
title = "Diarização de fala com whisper e ECAPA-TDNN"
demo = gr.Blocks(title=title)
demo.encrypt = False


with demo:
    with gr.Tab("Diarização do áurio de vídeo"):
        with gr.Row():
            gr.Markdown('''
                ### Cole um link para vídeo do youtube ou teste os exemplos:
                ''')
        examples = gr.Examples(examples=["https://www.youtube.com/watch?v=Ce1QWmbU8Eo",
                                         "https://www.youtube.com/watch?v=23GuG3JFvPY",
                                         "https://www.youtube.com/watch?v=p4QmQqSf1l8&t=1s",
                                         "https://www.youtube.com/watch?v=-BsAJyoj_Kw&t=72s"],
                               label="Exemplos", inputs=[youtube_url_in])

        with gr.Row():
            with gr.Column():
                youtube_url_in.render()
                download_youtube_btn = gr.Button("Obter vídeo do youtube")
                download_youtube_btn.click(get_youtube, [youtube_url_in], [
                    video_in])
                print(video_in)

        with gr.Row():
            with gr.Column():
                video_in.render()
                with gr.Column():
                    gr.Markdown('''
                    \n
                    \n
                    ### Configure a transcrição.
                    ''')
                selected_source_lang.render()
                selected_whisper_model.render()
                number_speakers.render()
                transcribe_btn = gr.Button("Transcrever e separar os falantes")
                transcribe_btn.click(speech_to_text, [
                                     video_in, selected_source_lang, selected_whisper_model, number_speakers], [transcription_df, system_info])

        with gr.Row():
            gr.Markdown('''
            ##### Resultado da transcrição e a separação dos falantes.
            ##### ''')

        with gr.Row():
            with gr.Column():
                transcription_df.render()
                system_info.render()
                gr.Markdown(
                    '''
                    <center><a href="https://opensource.org/licenses/Apache-2.0"><img src='https://img.shields.io/badge/License-Apache_2.0-blue.svg' alt='License: Apache 2.0'></center>
                    '''
                )

    with gr.Tab("Diarização de áudo do microfone ou de arquivo"):
        gr.Markdown('''
              <div>
              <h1 style='text-align: center'>Microfone ou arquivo de áudio</h1>
              </div>
              Grave áudio no microfone ou carregue um arquivo.
          ''')
        microphone = gr.inputs.Audio(
            source="microphone", type="filepath", optional=True)
        upload = gr.inputs.Audio(
            source="upload", type="filepath", optional=True)
        transcribe_btn = gr.Button("Transcrever")
        text_output = gr.Textbox()
        with gr.Row():
            gr.Markdown('''
                ### Você pode testar os exemplos abaixo ou carregar um arquivo de áudio:
                ''')
        examples = gr.Examples(examples=["poincarre_klein.wav",
                                         "jose_ferreira.wav",
                                         ],
                               label="Exemplos", inputs=[upload])
        transcribe_btn.click(
            transcribe, [microphone, upload], outputs=text_output)

demo.launch(debug=True)
