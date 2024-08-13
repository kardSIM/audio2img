from transformers import SeamlessM4TFeatureExtractor, Wav2Vec2BertModel
from diffusers import StableDiffusionPipeline
import torch
import gradio as gr

from pipeline import AudioToImagePipeline

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pipeline = AudioToImagePipeline(
    audio_encoder=Wav2Vec2BertModel.from_pretrained('youzarsif/wav2vec2bert_2_diffusion'),
    feature_extractor=SeamlessM4TFeatureExtractor.from_pretrained('youzarsif/wav2vec2bert_2_diffusion', stride=2),
    diffusion_pipeline=StableDiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2-1"),
    device=device
)

def generate_image(audio_file):
    image = pipeline(audio_file)
    return image

demo = gr.Interface(
    fn=generate_image,
    inputs=gr.Audio(type="filepath"),
    outputs="image",
    title="Audio to Image Generation"
)

demo.launch()