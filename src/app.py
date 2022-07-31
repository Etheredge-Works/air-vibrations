from string import punctuation
import gradio as gr
from transformers import pipeline
from transformers import AutomaticSpeechRecognitionPipeline
from deepmultilingualpunctuation import PunctuationModel

puntuation_model = PunctuationModel()
# capitalization_model = ("KES/caribe-capitalise")
# text = "My name is Clara and I live in Berkeley California Ist das eine Frage Frau Müller"
# print(result)
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModel

description = """
# Gradio Demo for exploring Speech Transcription.

Upload an audio file or record yourself to see a transcription.
The transcription passes through 4 models: transcription, punctuation, capitalization, and summarization. 
All output is given

Tips:
- Large files will take a while to process.
- Live recording is on the second tab.
"""


capitalise_tokenizer = AutoTokenizer.from_pretrained("KES/caribe-capitalise")
capitalise_model = AutoModelForSeq2SeqLM.from_pretrained("KES/caribe-capitalise")
spell_tokenizer = AutoTokenizer.from_pretrained("murali1996/bert-base-cased-spell-correction")
spell_model = AutoModel.from_pretrained("murali1996/bert-base-cased-spell-correction")

summarizer = pipeline("summarization")

pipe = pipeline(
    model="facebook/wav2vec2-large-960h", 
    chunk_length_s=90,
    stride_length_s=15,
)

def translate(audio_file):
    x = pipe(audio_file)
    text = x['text']
    return text


def punctuation(text):
    punctuation = puntuation_model.restore_punctuation(text)
    return punctuation


def capitalise(text):
    text = text.lower()
    inputs = capitalise_tokenizer("text:"+text, truncation=True, return_tensors='pt')
    # print(capitalization)
    output = capitalise_model.generate(inputs['input_ids'], num_beams=4, max_length=4096, early_stopping=True)
    # output = capitalise_model.generate(inputs['input_ids'], num_beams=4, max_length=1024, early_stopping=True)
    capitalised_text = capitalise_tokenizer.batch_decode(output, skip_special_tokens=True)

    result = ("".join(capitalised_text))
    return result


def spell_check(text):
    text = text.lower()
    inputs = spell_tokenizer(text, return_tensors='pt')
    # print(capitalization)
    output = spell_model.generate(inputs)
    spell_text = spell_tokenizer.batch_decode(output, skip_special_tokens=True)

    result = ("".join(spell_text))

    return result


def summarize(text):
    results = None
    length = len(text)
    while not results:
        try:
            results = summarizer(text[:length], min_length=10, max_length=128)
        except IndexError:
            print(f"shortening text: {length} -> {length//2}")
            length = length // 2
    return results[0]['summary_text']

def all(file):
    trans_text = translate(file).lower()
    punct_text = punctuation(trans_text)
    cap_text = capitalise(punct_text)
    sum_text = summarize(punct_text)
    return trans_text, punct_text, cap_text, sum_text

input = gr.Audio(type="filepath")
live_in = gr.Audio(type="filepath", source="microphone")
# options = gr.CheckboxGroup(
#     options=["text", "punctuation", "capitalisation"],
# )
raw_output = gr.Text(label="Raw Output")
puncuation_output = gr.Text(label="Punctuation Output")
capitalization_output = gr.Text(label="Capitalization Output")
sum_output = gr.Text(label="Summarized Output")

# translater = gr.Interface(
#     fn=translate, 
#     inputs=input, 
#     outputs=raw_output)

# punctuation = gr.Interface(
#     fn=punctuation,
#     inputs=raw_output,
#     outputs=puncuation_output)

# capitalization = gr.Interface(
#     fn=capitalise,
#     inputs=puncuation_output,
#     outputs=capitalization_output]



# gr.Series(translater, punctuation, capitalization).launch(share=True)
live_demo = gr.Interface(
    fn=all,
    inputs=live_in,
    outputs=[raw_output, puncuation_output, capitalization_output, sum_output],
    description=description)
demo = gr.Interface(
    fn=all,
    inputs=input,
    outputs=[raw_output, puncuation_output, capitalization_output, sum_output],
    description=description)

# interface = gr.Series(
#     gr.Textbox(value=description, show_label=False, interactive=False),
#     gr.TabbedInterface([demo, live_demo], tab_names=["Upload File", "Record Self"])
# )
interface = gr.TabbedInterface([demo, live_demo], tab_names=["Upload File", "Record Self"])
interface.launch()