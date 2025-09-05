import numpy as np


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