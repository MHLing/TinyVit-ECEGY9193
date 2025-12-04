import matplotlib.pyplot as plt

# 数据
names = [
    "layers.1.blocks.0",
    "layers.1.blocks.1",
    "layers.2.blocks.0",
    "layers.2.blocks.1",
    "layers.2.blocks.2",
    "layers.2.blocks.3",
    "layers.2.blocks.4",
    "layers.2.blocks.5",
    "layers.3.blocks.0",
    "layers.3.blocks.1",
]

values = [
    1.7455,
    3.7225,
    2.3205,
    2.6932,
    2.6484,
    2.8598,
    2.7929,
    4.3976,
    4.9253,
    3.6167,
]

x = list(range(len(names)))

plt.figure(figsize=(10, 4))
plt.plot(x, values, marker="o")
plt.xticks(x, names, rotation=45, ha="right")
plt.xlabel("Blocks")
plt.ylabel("Entropy")
plt.title("Entropy per TinyViT Block")
plt.tight_layout()
plt.grid(True, linestyle="--", alpha=0.4)

plt.savefig("entropy_blocks.png", dpi=300)
plt.show()
