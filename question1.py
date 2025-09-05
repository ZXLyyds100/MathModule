import numpy as np
import matplotlib.pyplot as plt

def trajectory_missile_A(t, missile_init, v_missile=300, target_fake=np.array([0, 0, 0])):
    """计算《A题.pdf》导弹t时刻位置（匀速直线直指假目标，6-6）"""
    dir_vec = target_fake - missile_init
    dir_norm = np.linalg.norm(dir_vec)
    if t > dir_norm / v_missile:  # 导弹到达假目标后静止
        return target_fake
    dir_unit = dir_vec / dir_norm
    return missile_init + v_missile * t * dir_unit


def trajectory_uav_A(t, uav_init, v_uav, theta_uav):
    """计算《A题.pdf》无人机t时刻位置（等高度匀速，6-7）"""
    theta_rad = np.radians(theta_uav)
    dx = v_uav * np.cos(theta_rad) * t  # x方向位移

    dy = v_uav * np.sin(theta_rad) * t  # y方向位移
    return uav_init + np.array([dx, dy, 0])  # 等高度飞行，z坐标不变


def trajectory_smoke_A(t, t_drop, t_delay, uav_init, v_uav, theta_uav, g=9.8, v_sink=3):
    """计算《A题.pdf》烟幕云团t时刻位置（自由下落+匀速下沉，6-5）"""
    t_det = t_drop + t_delay  # 起爆时刻
    theta_rad = np.radians(theta_uav)

    if t < t_drop:
        return np.array([np.nan, np.nan, np.nan])  # 未投放，位置无效
    elif t_drop <= t < t_det:
        # 投放后→起爆前：自由下落
        t_free = t - t_drop
        x = uav_init[0] + v_uav * np.cos(theta_rad) * t
        y = uav_init[1] + v_uav * np.sin(theta_rad) * t
        z = uav_init[2] - 0.5 * g * t_free ** 2  # 自由下落公式
        return np.array([x, y, z])
    else:
        # 起爆后：匀速下沉
        t_sink = t - t_det
        # 计算起爆点位置
        x_det = uav_init[0] + v_uav * np.cos(theta_rad) * t_det
        y_det = uav_init[1] + v_uav * np.sin(theta_rad) * t_det
        z_det = uav_init[2] - 0.5 * g * t_delay ** 2
        z = z_det - v_sink * t_sink  # 匀速下沉
        return np.array([x_det, y_det, z])


def generate_target_points_A(center=np.array([0, 200, 0]), r=7, h=10, n=5):
    """生成《A题.pdf》真目标采样点（5×5×5=125个，6-6）"""
    target_points = []
    for z in np.linspace(0, h, n):  # 高度方向：0~10m，5个点
        for rad in np.linspace(0, r, n):  # 半径方向：0~7m，5个点
            for theta in np.linspace(0, 2 * np.pi, n):  # 圆周方向：0~2π，5个点
                x = center[0] + rad * np.cos(theta)
                y = center[1] + rad * np.sin(theta)
                target_points.append(np.array([x, y, z]))
    return np.array(target_points)


def is_shielded_A(t, missile_pos, smoke_pos, target_points, r_smoke=10):
    """《A题.pdf》遮蔽判定：导弹-真目标连线是否穿烟幕（6-5）"""
    for tp in target_points:
        AB = tp - missile_pos  # 导弹→真目标采样点向量
        AC = smoke_pos - missile_pos  # 导弹→烟幕中心向量
        AB_sq = np.dot(AB, AB)
        if AB_sq < 1e-6:  # 避免导弹与采样点重合的计算错误
            continue
        # 线段-球相交的二次方程系数
        a = AB_sq
        b = -2 * np.dot(AC, AB)
        c = np.dot(AC, AC) - r_smoke ** 2
        discriminant = b ** 2 - 4 * a * c

        if discriminant < 0:  # 无实根，不相交
            continue
        # 求解二次方程，判断根是否在[0,1]内
        s1 = (-b - np.sqrt(discriminant)) / (2 * a)
        s2 = (-b + np.sqrt(discriminant)) / (2 * a)
        if (0 <= s1 <= 1) or (0 <= s2 <= 1):
            return True  # 遮蔽有效
    return False  # 遮蔽无效
# --------------------------
# 1. 数据准备（来自《A题.pdf》小问1固定参数）
# --------------------------
uav_init = np.array([17800, 0, 1800])  # FY1初始位置
v_uav = 120  # FY1速度
theta_uav = 180  # FY1方向（朝假目标）
t_drop = 1.5  # 投放时间
t_delay = 3.6  # 起爆延迟
missile_init = np.array([20000, 0, 2000])  # M1初始位置
target_fake = np.array([0, 0, 0])  # 假目标
t_det = t_drop + t_delay  # 起爆时刻
t_missile_arrive = np.linalg.norm(missile_init - target_fake) / 300  # M1到达时间（≈67s）
target_points = generate_target_points_A()  # 真目标采样点

# 生成各主体的轨迹数据
# 无人机轨迹（t∈[0, t_drop+1]，覆盖投放前）
t_uav = np.arange(0, t_drop + 1, 0.5)
uav_trajs = [trajectory_uav_A(t, uav_init, v_uav, theta_uav) for t in t_uav]
# 导弹轨迹（t∈[0, 30]，覆盖烟幕有效时段）
t_missile = np.arange(0, 30, 0.5)
missile_trajs = [trajectory_missile_A(t, missile_init) for t in t_missile]
# 烟幕轨迹（t∈[t_det, t_det+20]，覆盖有效时间）
t_smoke = np.arange(t_det, t_det + 20, 0.5)
smoke_trajs = [trajectory_smoke_A(t, t_drop, t_delay, uav_init, v_uav, theta_uav) for t in t_smoke]

# --------------------------
# 2. 绘制3D轨迹图
# --------------------------
plt.rcParams['font.sans-serif'] = ['SimHei']  # 解决中文显示
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示
fig = plt.figure(figsize=(12, 8))  # 图大小（适配论文页面）
ax = fig.add_subplot(111, projection='3d')

# 绘制无人机轨迹
ax.plot(
    [p[0] for p in uav_trajs],
    [p[1] for p in uav_trajs],
    [p[2] for p in uav_trajs],
    'b-', linewidth=2, label='FY1轨迹（朝假目标）'
)
# 标注无人机投放点
drop_pos = trajectory_uav_A(t_drop, uav_init, v_uav, theta_uav)
ax.scatter(drop_pos[0], drop_pos[1], drop_pos[2], c='blue', s=80, label='烟幕投放点', marker='^')

# 绘制导弹轨迹
ax.plot(
    [p[0] for p in missile_trajs],
    [p[1] for p in missile_trajs],
    [p[2] for p in missile_trajs],
    'r-', linewidth=2, label='M1轨迹（直指假目标）'
)
ax.scatter(missile_init[0], missile_init[1], missile_init[2], c='red', s=80, label='M1初始位置')

# 绘制烟幕轨迹
ax.plot(
    [p[0] for p in smoke_trajs],
    [p[1] for p in smoke_trajs],
    [p[2] for p in smoke_trajs],
    'g-', linewidth=2, label='烟幕云团轨迹（匀速下沉）'
)
# 标注烟幕起爆点
det_pos = trajectory_smoke_A(t_det, t_drop, t_delay, uav_init, v_uav, theta_uav)
ax.scatter(det_pos[0], det_pos[1], det_pos[2], c='green', s=80, label='烟幕起爆点', marker='s')

# 绘制真目标（圆柱）
theta_cyl = np.linspace(0, 2*np.pi, 100)  # 圆周方向采样
# 真目标下底面（z=0）
ax.plot(
    0 + 7*np.cos(theta_cyl),
    200 + 7*np.sin(theta_cyl),
    np.zeros_like(theta_cyl),
    'k-', linewidth=2, label='真目标下底面（r=7m）'
)
# 真目标上底面（z=10m）
ax.plot(
    0 + 7*np.cos(theta_cyl),
    200 + 7*np.sin(theta_cyl),
    np.ones_like(theta_cyl)*10,
    'k--', linewidth=2, label='真目标上底面（h=10m）'
)

# 坐标轴标签与图例
ax.set_xlabel('X坐标 (m)', fontsize=11)
ax.set_ylabel('Y坐标 (m)', fontsize=11)
ax.set_zlabel('Z坐标 (m)', fontsize=11)
ax.set_title('图5-1 小问1：M1-FY1-烟幕3D轨迹图', fontsize=12, pad=20)
ax.legend(loc='upper right', fontsize=10)
ax.grid(True, alpha=0.3)  # 网格线增强可读性

# 保存图片（dpi=300确保清晰度，适配印刷）
plt.savefig('小问1_3D轨迹图.png', dpi=300, bbox_inches='tight')  # bbox_inches避免标签被截断
plt.close()