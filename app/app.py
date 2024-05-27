# 簡単なindexページの作成を行っています。
from flask import Flask, jsonify
import open_clip
import torch
from PIL import Image


app = Flask(__name__)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Cocaの読み込み
model, _, transform = open_clip.create_model_and_transforms(
    "coca_ViT-L-14",
    pretrained="laion2b_s13b_b90k",
    device=device,
)


img = Image.open("./images/test.jpg").convert("RGB")


@app.route("/")
def index():
    return "index page"


@app.route("/register")
def register():
    return jsonify({"message": "test"})


@app.route("/photo_test")
def photo_test():
    # 画像からキャプションを生成
    im = transform(img).unsqueeze(0).to(device)
    with torch.no_grad(), torch.cuda.amp.autocast():
        generated = model.generate(im, seq_len=20)

    # キャプションを人間が読める文章に変換して表示
    caption = (
        open_clip.decode(generated[0].detach())
        .split("<end_of_text>")[0]
        .replace("<start_of_text>", "")
    )
    print(caption)
    return jsonify({"message": caption})
