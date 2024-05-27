# 簡単なindexページの作成を行っています。
from flask import Flask, jsonify
import open_clip
import torch
from googletrans import Translator
from PIL import Image


app = Flask(__name__)
app.json.ensure_ascii = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model, preprocess, _ = open_clip.create_model_and_transforms(
    "coca_ViT-L-14",
    device=device,
)
model_path = "./model_folder/model.pth"
model.load_state_dict(torch.load(model_path, map_location=device))
# モデルを評価モードに設定
model.eval()

# 以降はモデルを使った推論が可能
img = Image.open("./images/test.jpg").convert("RGB")


@app.route("/")
def index():
    return "index page"


@app.route("/register")
def register():
    return jsonify({"message": "test"})


@app.route("/photo_test")
def photo_test():
    im = preprocess(img).unsqueeze(0).to(device)
    with torch.no_grad(), torch.cuda.amp.autocast():
        generated = model.generate(im, seq_len=20)

    # キャプションを人間が読める文章に変換して表示
    caption = (
        open_clip.decode(generated[0].detach())
        .split("<end_of_text>")[0]
        .replace("<start_of_text>", "")
    )

    # 翻訳
    translator = Translator()
    translated_text = translator.translate(caption, src="en", dest="ja").text

    print(caption, translated_text)

    response = jsonify({"message": translated_text})

    return response
