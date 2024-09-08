from glob import glob
import gradio as gr
import torch
import pandas as pd
from cer.entity_relationship import  generateEntRelationship
from text_classification.jutsu_classifier import JutsuPredictor
from utils.data_loader import load_subtitles
from theme_classifier.classifer import themeClassifier
from dotenv import load_dotenv
import os

load_dotenv()
device = 'cuda' if torch.cuda.is_available() else 'cpu' 

# Define your functions
def process_file(subs_path, save_path):
    
    df = load_subtitles(subs_path)
    print(df.head)
    return df

def get_theme(subs_df):
    # Initialize the classifier with a batch size of 20
    classifier = themeClassifier( batch_size=20)
    


    # Load the model
    classifier.load_model()

    # Get themes from subtitles
    classes = classifier.get_theme(subs_df.subtitles)

    # Plot the theme distribution
    plot = classifier.plot_themes()

    return plot


def plot_relationships(subs):
    generator = generateEntRelationship(subs)
    scripts = generator.get_scripts()
    subs_df = generator.generate_subs_df()
    named_entities = generator.get_named_entities()
    relationships = generator.generate_entity_relationship()
    html_plot = generator.plot()

    return html_plot

def classify_text():
    classifier = JutsuPredictor(model_path=classifier_model_path,
                                data_path=classifier_data_path,
                                 hugging_face_token= os.getenv('HUGGING_FACE_TOKEN'))
    outputs = classifier.classify_justsu()
    return outputs
    




def main():
    with gr.Blocks() as iface:
        with gr.Row():
            gr.HTML('<h1> Welcome to Classifier</h1>')
        
        with gr.Row():
            with gr.Column():
                subs_df = gr.DataFrame()
                subs_plot = gr.Plot(label="Theme Classification Plot")
                subs_graph = gr.HTML()

            with gr.Column():
                subs_path = gr.TextArea(value='Enter subtitles path', label="Subtitles",lines = 1)
                save_path = gr.TextArea(value='Enter save path here', label="Save Path",lines=1)
                
                with gr.Row():
                    df_button = gr.Button(value='Display DataFrame')
                    df_button.click(fn=process_file, inputs=[subs_path, save_path], outputs=subs_df)

                with gr.Row():
                    plot_button = gr.Button(value='PLOT THEME')
                    plot_button.click(fn=get_theme, inputs=subs_df, outputs=subs_plot)

                with gr.Row():
                    plot_button = gr.Button(value='PLOT RELATIONSHIPS')
                    plot_button.click(fn=plot_relationships, inputs=subs_path, outputs=subs_graph)

        with gr.Row():
            with gr.Column():
                classifier_model_path = gr.Textbox(label='Enter Model Path',lines = 1)
                classifier_data_path = gr.Textbox(label='Enter Data Path',lines = 1)
                text_for_classifier = gr.Textbox(label='Enter Text To Classify',lines = 1)
                result_classification = gr.TextArea(label='Result')
            
                classify_button = gr.Button(value='Classify')
                classify_button.click(fn=classify_text, inputs=[text_for_classifier,classifier_data_path,classifier_model_path], outputs= result_classification)

                
    iface.launch()

if __name__ == '__main__':
    main()
