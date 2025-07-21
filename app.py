import gradio as gr
from model_utils import process_video

def analyse(video):
    if video is None:
        return None, None
    out_video, csv_path = process_video(video)
    return out_video, csv_path

demo = gr.Interface(
    fn=analyse,
    inputs=gr.Video(label="Vidéo de sprint (MP4)", format="mp4"),
    outputs=[gr.Video(label="Vidéo annotée"), gr.File(label="CSV des keypoints")],
    title="Sprint Pose Analysis (RTMPose + RTMDet)",
    description=("Chargez une courte vidéo de sprint (≤ 30 s). "
                 "RTMDet détecte la personne, RTMPose prédit la pose, "
                 "et vous récupérez la vidéo annotée + le CSV des keypoints.")
)

if __name__ == "__main__":
    demo.launch()
