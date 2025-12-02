import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt

from models.tiny_vit import tiny_vit_5m_224


def load_model(prune_ratio=0.2):
    model = tiny_vit_5m_224(
        pretrained=False,              # 你有预训练就改成 True
        token_pruning_ratio=prune_ratio,
        token_pruning_method='magnitude',
    )
    model.eval()
    return model


def preprocess(img_path):
    img = Image.open(img_path).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])
    x = transform(img).unsqueeze(0)  # [1,3,224,224]
    return img, x


@torch.no_grad()
def get_stage3_tokens_and_importance(model, x):
    # 1) patch embed
    x = model.patch_embed(x)  # [B,C,H,W]  H=W=56

    # 2) Stage1
    x = model.layers[0](x)    # ConvLayer -> PatchMerging, 输出形状 [B, L1, C1]

    # 3) Stage2
    x = model.layers[1](x)    # BasicLayer Stage2, 输出 [B, L2, C2]，这是 Stage3 的输入

    x_stage3_in = x

    # 4) 取 Stage3 的第一个 block
    stage3 = model.layers[2]           # BasicLayer
    block0 = stage3.blocks[0]          # 第一个 TinyViTBlock

    # 5) 直接用 block0 的 compute_token_importance
    importance = block0.compute_token_importance(x_stage3_in)  # [B, N]
    B, N = importance.shape

    H, W = stage3.input_resolution    # 正常是 (14,14)
    assert H * W == N, f"H*W={H*W}, N={N}"

    importance_map = importance[0].reshape(H, W).cpu()  # [H,W]
    return x_stage3_in, importance_map


def visualize(img, importance_map):
    # upsample 到 224x224
    H, W = importance_map.shape
    mask = importance_map.unsqueeze(0).unsqueeze(0)  # [1,1,H,W]
    mask = F.interpolate(mask, size=(224, 224), mode='bilinear', align_corners=False)
    mask = mask[0, 0].cpu().numpy()

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title("Original")
    plt.axis("off")
    plt.imshow(img)

    plt.subplot(1, 2, 2)
    plt.title("Token importance heatmap")
    plt.axis("off")
    plt.imshow(img)
    plt.imshow(mask, alpha=0.5)  # 叠加热力图
    plt.show()


if __name__ == "__main__":
    img_path = "your_image.jpg"  # 换成你的图片路径
    img, x = preprocess(img_path)
    model = load_model(prune_ratio=0.2)

    _, importance_map = get_stage3_tokens_and_importance(model, x)
    visualize(img, importance_map)
