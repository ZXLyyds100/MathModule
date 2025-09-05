import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
# 导入论文中定义的核心函数（需确保路径正确）
from core_functions import *
# --------------------------
# 1. 数据准备（严格匹配《A题.pdf》小问1固定参数）
# --------------------------
uav_init = np.array([17800, 0, 1800])  # FY1初始位置（A题6-6）
v_uav = 120  # FY1速度（A题小问1固定）
theta_uav = 180  # FY1方向（朝假目标，x负向）
t_drop = 1.5  # 投放时间（A题小问1固定）
t_delay = 3.6  # 起爆延迟（A题小问1固定）
missile_init = np.array([20000, 0, 2000])  # M1初始位置（A题6-6）
target_fake = np.array([0, 0, 0])  # 假目标（原点）
t_det = t_drop + t_delay  # 起爆时刻（5.1s）
t_missile_arrive = np.linalg.norm(missile_init - target_fake) / 300  # M1到达时间（≈67s）
target_points = generate_target_points_A()  # 真目标125个采样点

# 生成各主体轨迹数据（时间步长0.5s，平衡精度与流畅度）
# 无人机轨迹（覆盖投放前：t∈[0, t_drop+1]）
t_uav = np.arange(0, t_drop + 1, 0.5)
uav_trajs = [trajectory_uav_A(t, uav_init, v_uav, theta_uav) for t in t_uav]
# 导弹轨迹（覆盖烟幕有效时段：t∈[0, 30]）
t_missile = np.arange(0, 30, 0.5)
missile_trajs = [trajectory_missile_A(t, missile_init) for t in t_missile]
# 烟幕轨迹（覆盖有效时间：t∈[t_det, t_det+20]）
t_smoke = np.arange(t_det, t_det + 20, 0.5)
smoke_trajs = [trajectory_smoke_A(t, t_drop, t_delay, uav_init, v_uav, theta_uav) for t in t_smoke]

# --------------------------
# 2. 绘图配置（解决中文乱码，确保清晰度）
# --------------------------
plt.rcParams['font.sans-serif'] = ['SimHei']  # 中文显示
plt.rcParams['axes.unicode_minus'] = False  # 负号显示
fig = plt.figure(figsize=(12, 8))  # 图大小（适配A4纸，横向更适合3D图）
ax = fig.add_subplot(111, projection='3d')

# --------------------------
# 3. 绘制各主体轨迹与关键点位
# --------------------------
# ① 无人机轨迹（蓝色）
ax.plot(
    [p[0] for p in uav_trajs],
    [p[1] for p in uav_trajs],
    [p[2] for p in uav_trajs],
    'b-', linewidth=2.5, label='FY1轨迹（朝假目标飞行）'
)
# 标注无人机投放点（三角形）
drop_pos = trajectory_uav_A(t_drop, uav_init, v_uav, theta_uav)
ax.scatter(
    drop_pos[0], drop_pos[1], drop_pos[2],
    c='darkblue', s=100, label='烟幕投放点', marker='^', edgecolors='black'
)

# ② 导弹轨迹（红色）
ax.plot(
    [p[0] for p in missile_trajs],
    [p[1] for p in missile_trajs],
    [p[2] for p in missile_trajs],
    'r-', linewidth=2.5, label='M1轨迹（直指假目标）'
)
# 标注导弹初始位置（圆形）
ax.scatter(
    missile_init[0], missile_init[1], missile_init[2],
    c='darkred', s=100, label='M1初始位置', marker='o', edgecolors='black'
)

# ③ 烟幕轨迹（绿色）
ax.plot(
    [p[0] for p in smoke_trajs],
    [p[1] for p in smoke_trajs],
    [p[2] for p in smoke_trajs],
    'g-', linewidth=2.5, label='烟幕云团轨迹（匀速下沉）'
)
# 标注烟幕起爆点（方形）
det_pos = trajectory_smoke_A(t_det, t_drop, t_delay, uav_init, v_uav, theta_uav)
ax.scatter(
    det_pos[0], det_pos[1], det_pos[2],
    c='darkgreen', s=100, label='烟幕起爆点', marker='s', edgecolors='black'
)

# ④ 真目标（黑色圆柱，覆盖上下底面）
theta_cyl = np.linspace(0, 2*np.pi, 100)  # 圆周方向采样100点，确保圆柱光滑
# 下底面（z=0，真目标下底面圆心(0,200,0)）
ax.plot(
    0 + 7*np.cos(theta_cyl),
    200 + 7*np.sin(theta_cyl),
    np.zeros_like(theta_cyl),
    'k-', linewidth=2, label='真目标下底面（r=7m）'
)
# 上底面（z=10m，真目标高度10m）
ax.plot(
    0 + 7*np.cos(theta_cyl),
    200 + 7*np.sin(theta_cyl),
    np.ones_like(theta_cyl)*10,
    'k--', linewidth=2, label='真目标上底面（h=10m）'
)

# --------------------------
# 4. 坐标轴与图例配置（确保评委易读）
# --------------------------
ax.set_xlabel('X坐标 (m)', fontsize=12, labelpad=10)
ax.set_ylabel('Y坐标 (m)', fontsize=12, labelpad=10)
ax.set_zlabel('Z坐标 (m)', fontsize=12, labelpad=10)
ax.set_title('图5-1 小问1：M1-FY1-烟幕3D轨迹图', fontsize=14, pad=20)
ax.legend(loc='upper right', fontsize=11, framealpha=0.9)  # 图例半透明，不遮挡轨迹
ax.grid(True, alpha=0.3)  # 网格线增强空间感

# --------------------------
# 5. 保存图片（印刷级清晰度，避免标签截断）
# --------------------------
plt.savefig('小问1_3D轨迹图.png', dpi=300, bbox_inches='tight')  # bbox_inches解决标签被截断
plt.close()