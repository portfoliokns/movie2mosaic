# このリポジトリについて
このリポジトリは動画に移る人の顔にモザイクを付け加えるためのリポジトリになります。動画を撮影した際、どうしても写ってしまう無関係な人の顔にモザイクを自動的に変換することができます。

# 準備
## インストール
### Python
このリポジトリを使用するためには、あらかじめPythonがPCに入っていないといけません。そのため、あらかじめインストールしておいてください。なお開発段階では"Python 3.10.11"を使用していました。（バージョンによって互換性があるかもしれません）

```
pip install face_recognition opencv-python moviepy tqdm
```

### face-recognition: 顔認識ライブラリ
```
pip install face_recognition
```

### opencv-python: OpenCV（画像やビデオの処理に使用）
```
pip install opencv-python
```

### moviepy: 動画の編集に使用
```
pip install moviepy
```

### tqdm: 進捗バー表示ライブラリ
```
pip install tqdm
```

# 使い方
## GitHubからクローン
GitHub上からこのリポジトリを任意のフォルダにクローンしてください。

## 変換したい動画の用意
クローンしたリポジトリと同じ階層に動画を配置してください。

## 各種パラメータを設定
ソースコードの以下の部分を設定してください。
```
input_video = "./test.mov"
output_video = 'output_video.mp4'
final_output = 'output_video_audio.mp4'
mosaic_para = 8
```

## movie2mosaic.pyを実行
ターミナル上から"movie2mosaic.py"を実行して、動画を変換してください。
```
python movie2mosaic.py
```

## 変換終了後
クローンしたリポジトリと同じ階層にモザイク加工された動画ファイルが保存されています。

