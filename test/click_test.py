import gradio as gr


def onSelectImg(img):
    return img

with gr.Blocks(theme=gr.themes.Monochrome()) as app:
    with gr.Column(align="center"):
        image = gr.Image()
        output = gr.Image()
    image.select(
            fn=onSelectImg,
            inputs=[image],
            outputs=[output],
            )



app.launch(debug=True)