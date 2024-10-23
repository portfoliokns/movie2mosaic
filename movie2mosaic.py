import face_recognition as recognition
import cv2 as cv2
from moviepy.editor import VideoFileClip
from tqdm import tqdm

# モザイク処理
def mosaic(img, x, y, w, h, size):
    (x1, y1, x2, y2) = (x, y, x+w, y+h)
    img_rec = img[y1:y2, x1:x2]
    img_small = cv2.resize(img_rec, (size, size))
    img_mos = cv2.resize(img_small, (w, h), interpolation=cv2.INTER_AREA)
    img_out = img.copy()
    img_out[y1:y2, x1:x2] = img_mos
    return img_out

# 各種パラメータ
input_video = "./test.mov"  #モザイク処理したい動画
output_video = 'output_video.mp4'  #出力ファイル名（音声なし）（出力形式は.mp4にしてください）
final_output = 'output_video_audio.mp4'  #出力ファイル名（音声あり）（出力形式は.mp4にしてください）
mosaic_para = 8  #モザイクパラメータ（値を大きくすると、人の顔が判別しやすくなります）

# 動画の読み込み
cap = cv2.VideoCapture(input_video)

print("モザイク処理を開始します")

bar = tqdm()

# モザイク処理
frames = []
while True:
    ret, frame = cap.read()
    if not ret:
        break

    bar.update(1)

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_locations = recognition.face_locations(rgb_frame)
    image_cv2 = frame.copy()

    for (top, right, bottom, left) in face_locations:
        x, y, w, h = left, top, right - left, bottom - top
        image_cv2 = mosaic(image_cv2, x, y, w, h, mosaic_para)

    frames.append(image_cv2)

print("モザイク処理が終了しました")

# フレームをまとめて動画化する
print("動画への変換を開始します")
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
out = cv2.VideoWriter(output_video, fourcc, fps, (frame_width, frame_height))

for frame in tqdm(frames):
    out.write(frame)

cap.release()
out.release()
cv2.destroyAllWindows()
print("動画への変換が終了しました")

# 音声の追加(MoviePyを使って音声を抽出し、映像に結合)
original_video = VideoFileClip(input_video)
audio_clip = original_video.audio  # 元の動画から音声を抽出
new_video = VideoFileClip(output_video)  # OpenCVで作成した動画を読み込み

# 音声付きで新しい動画を書き出す
final_video = new_video.set_audio(audio_clip)
final_video.write_videofile(final_output, codec='libx264', audio_codec='aac')