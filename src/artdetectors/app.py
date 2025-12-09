
import gradio as gr
from artdetectors.pipeline import ImageAnalysisPipeline
from PIL import Image
import torch

import os
os.environ["TRANSFORMERS_NO_TF"] = "1"

# Load once at startup
pipeline = ImageAnalysisPipeline()

def analyze_image(image):
    if image is None:
        return "No image uploaded.", None, None

    pil_image = Image.fromarray(image).convert("RGB")

    result = pipeline.analyze(pil_image)

    # Format “styles” as top-k list
    styles_formatted = "\n".join(
        [f"{name}: {score:.4f}" for name, score in result["styles"]]
    )

    # Format SuSy predictions nicely
    susy_classes = result["susy"]["probs"]
    susy_formatted_list = [
        f"{k}: {v:.4f}" for k, v in susy_classes.items()
    ]
    susy_formatted = "\n".join(susy_formatted_list)

    caption = result["caption"]

    return caption, styles_formatted, susy_formatted


# Build the interface
with gr.Blocks() as demo:
    gr.Markdown("# MLAngelo — Style, Caption, Model Detection")

    with gr.Row():
        image_input = gr.Image(type="numpy", label="Upload artwork")
    
    analyze_btn = gr.Button("Analyze")
    
    with gr.Row():
        caption_out = gr.Textbox(label="BLIP Caption")
        styles_out = gr.Textbox(label="Top Predicted Styles")
        susy_out = gr.Textbox(label="Source (SuSy Probabilities)")

    analyze_btn.click(
        fn=analyze_image,
        inputs=[image_input],
        outputs=[caption_out, styles_out, susy_out]
    )

demo.launch()
