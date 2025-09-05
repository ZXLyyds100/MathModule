import numpy as np
import matplotlib.pyplot as plt
from core_functions import (trajectory_missile_A, trajectory_smoke_A,
                          generate_target_points_A, is_shielded_A)

# --------------------------
# 1. 数据准备（与3D轨迹图一致，确保结果连贯）
# --------------------------
uav_init = np.array([17800, 0, 1800])
v_uav = 120
theta_uav = 180
t_drop = 1.5
t_delay = 3.6
missile_init = np.array([20000, 0, 2000])
target_fake = np.array([0, 0, 0])
t_det = t_drop + t_delay
t_missile_arrive = np.linalg.norm(missile_init - target_fake) / 300
target_points = generate_target_points_A()
r_smoke = 10  # 烟幕有效半径（A题6-5固定）

# 计算每个时刻的遮蔽状态（时间步长0.1s，确保时段精度）
t_total = np.arange(0, t_missile_arrive, 0.1)  # 覆盖导弹飞行全程
shielded = np.zeros_like(t_total)  # 1=有效，0=无效
smoke_valid = np.zeros_like(t_total)  # 1=烟幕物理有效（起爆后20s）

for i, t in enumerate(t_total):
    # 判断烟幕是否物理有效
    if t_det <= t <= t_det + 20:
        smoke_valid[i] = 1
    # 计算遮蔽状态
    if smoke_valid[i] == 1:
        missile_pos = trajectory_missile_A(t, missile_init)
        smoke_pos = trajectory_smoke_A(t, t_drop, t_delay, uav_init, v_uav, theta_uav)
        if not np.any(np.isnan(smoke_pos)):
            if is_shielded_A(t, missile_pos, smoke_pos, target_points, r_smoke):
                shielded[i] = 1

# --------------------------
# 2. 绘图配置
# --------------------------
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
fig = plt.figure(figsize=(14, 5))  # 横向图，适合时间轴

# --------------------------
# 3. 绘制时间轴
# --------------------------
# ① 烟幕物理有效范围（橙色阴影，半透明）
plt.fill_between(
    t_total, 0, smoke_valid,
    color='orange', alpha=0.3, label='烟幕物理有效范围（起爆后20s）'
)
# ② 有效遮蔽时段（绿色阴影，半透明）
plt.fill_between(
    t_total, 0, shielded,
    color='green', alpha=0.5, label='有效遮蔽时段'
)
# ③ 导弹到达时间（红色虚线，标注关键节点）
plt.axvline(
    x=t_missile_arrive, color='red', linestyle='--', linewidth=2,
    label=f'M1到达假目标时间（{t_missile_arrive:.1f}s）'
)
# ④ 烟幕起爆时间（蓝色虚线，标注关键节点）
plt.axvline(
    x=t_det, color='blue', linestyle='--', linewidth=2,
    label=f'烟幕起爆时间（{t_det:.1f}s）'
)

# --------------------------
# 4. 坐标轴与标注（量化关键指标）
# --------------------------
plt.xlabel('时间 (s)', fontsize=12)
plt.ylabel('状态（1=有效，0=无效）', fontsize=12)
plt.title('图5-2 小问1：烟幕有效遮蔽时间轴图', fontsize=14, pad=15)
plt.ylim(-0.1, 1.1)  # 上下留空，避免标签紧贴边界
plt.xlim(0, 40)  # 聚焦烟幕有效时段（0~40s），避免后期无效时段干扰
plt.legend(fontsize=11, loc='upper right')
plt.grid(True, alpha=0.3, axis='x')  # 仅x轴网格，增强时间可读性

# 标注有效遮蔽时长（核心结果，让评委一眼看到）
effective_duration = np.sum(shielded) * 0.1  # 步长0.1s，总时长=有效点数×0.1
plt.text(
    30, 0.5,  # 文字位置（右上角，不遮挡关键区域）
    f'有效遮蔽时长：{effective_duration:.2f}s\n占烟幕物理有效时间比例：{effective_duration/20*100:.1f}%',
    fontsize=11, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
    verticalalignment='center'
)

# --------------------------
# 5. 保存图片
# --------------------------
plt.savefig('小问1_时间轴图.png', dpi=300, bbox_inches='tight')
plt.close()