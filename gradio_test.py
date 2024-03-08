import gradio as gr
import os


def video_identity(video):
    return video


demo = gr.Interface(video_identity, 
                    gr.Video(), 
                    "playable_video")

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=6002)
