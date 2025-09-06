import numpy as np
import matplotlib.pyplot as plt

# --------------------------
# 1. 数据准备（严格匹配《A题.pdf》小问5弹数结果）
# --------------------------
# 无人机弹数分配（无人机编号：弹数）
uav_missile_count = {
    "FY1": 2,
    "FY2": 1,
    "FY3": 2,
    "FY4": 1,
    "FY5": 2
}
# 统计弹数分类（1枚/2枚）
count_category = {}
for uav, count in uav_missile_count.items():
    key = f"{count}枚弹"
    if key not in count_category:
        count_category[key] = {"count": 0, "uavs": []}
    count_category[key]["count"] += 1
    count_category[key]["uavs"].append(uav)

# 提取饼图数据
labels = list(count_category.keys())  # 标签：["1枚弹", "2枚弹"]
sizes = [count_category[label]["count"] for label in labels]  # 数量：[2, 3]
uav_labels = [f"{label}\n（{', '.join(count_category[label]['uavs'])}）" for label in labels]  # 标注无人机
colors = ["#ff9999", "#66b3ff"]  # 区分颜色
explode = (0.05, 0.05)  # 轻微突出，提升可读性

# --------------------------
# 2. 绘图配置（符合论文格式）
# --------------------------
plt.rcParams['font.sans-serif'] = ['SimHei']  # 中文显示
plt.rcParams['axes.unicode_minus'] = False    # 负号显示
fig, ax = plt.subplots(figsize=(8, 8))  # 圆形饼图，比例1:1

# --------------------------
# 3. 绘制饼图
# --------------------------
wedges, texts, autotexts = ax.pie(
    sizes,
    explode=explode,
    labels=uav_labels,
    colors=colors,
    autopct='%1.1f%%',  # 显示占比（1位小数）
    shadow=True,
    startangle=90,
    textprops={'fontsize': 12}  # 文本字号
)

# 美化占比文字（加粗、白色）
for autotext in autotexts:
    autotext.set_color('white')
    autotext.set_fontweight('bold')
    autotext.set_fontsize(11)

# --------------------------
# 4. 标题与注释（关联《A题.pdf》约束）
# --------------------------
ax.set_title(
    '图C.1 《A题.pdf》小问5：5架无人机烟幕弹数分布饼图',
    fontsize=14,
    pad=20
)
# 添加核心结论注释
ax.text(
    1.3, 0.5,  # 右侧文字位置
    f'核心结论：\n1. 弹数分布：1枚弹（2架，40%）、2枚弹（3架，60%）\n2. 所有无人机弹数≤3枚，符合《A题.pdf》6-13约束\n3. 无资源过载或闲置，分配均衡',
    transform=ax.transAxes,
    fontsize=11,
    bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8),
    verticalalignment='center'
)

# --------------------------
# 5. 保存图片（印刷级清晰度）
# --------------------------
plt.tight_layout()
plt.savefig('《A题.pdf》附录C1_小问5弹数分布饼图.png', dpi=300, bbox_inches='tight')
plt.close()
print("C.1 饼图已保存：《A题.pdf》附录C1_小问5弹数分布饼图.png")