import open_clip
import torch
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
model, preprocess, _ = open_clip.create_model_and_transforms(
    "coca_ViT-L-14",
    device=device,
)
model_path = "./app/model_folder/model.pth"
model.load_state_dict(torch.load(model_path, map_location=device))
# モデルを評価モードに設定
model.eval()

# 以降はモデルを使った推論が可能
img = Image.open("./app/images/test.jpg").convert("RGB")

im = preprocess(img).unsqueeze(0).to(device)
with torch.no_grad(), torch.cuda.amp.autocast():
    generated = model.generate(im, seq_len=20)

# キャプションを人間が読める文章に変換して表示
caption = (
    open_clip.decode(generated[0].detach())
    .split("<end_of_text>")[0]
    .replace("<start_of_text>", "")
)

print(caption)
