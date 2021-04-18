import librosa
import os  
import fastaudio.core.signal as fcs

lvpath ="E:\Datasets\Voice\Librivox\dev\LibriSpeech\dev-clean"
mcvpath ="E:\Datasets\Voice\Mozilla Common Voice\en\cv-corpus-6.1-2020-12-11\en"

def get_file_transcript(file_path):
    file_name = str(file_path).split('\\')[-1]
    file_extension = file_name.split('.')[-1]
    folder_path = '\\'.join(str(file_path).split('\\')[:-1])
    transcript_file = next(x for x in os.listdir(folder_path) if x.endswith('.txt'))
    f = open(folder_path+'\\'+transcript_file, "r")
    file_trans = next(x for x in f.readlines() if x.startswith(file_name.removesuffix("."+file_extension)))
    file_trans= file_trans.removeprefix(file_name.removesuffix("."+file_extension)).removesuffix('\n').strip()
    return file_trans