from scipy.optimize import linear_sum_assignment
from deap import base, creator, tools, algorithms
import pandas as pd
import numpy as np
import openpyxl
import random

# ----------------- 轨迹与判定函数 -----------------
def trajectory_missile_A(t, missile_init, v_missile=300, target_fake=np.array([0, 0, 0])):
    dir_vec = target_fake - missile_init
    dir_norm = np.linalg.norm(dir_vec)
    if t > dir_norm / v_missile:
        return target_fake
    dir_unit = dir_vec / dir_norm
    return missile_init + v_missile * t * dir_unit

def trajectory_uav_A(t, uav_init, v_uav, theta_uav):
    theta_rad = np.radians(theta_uav)
    dx = v_uav * np.cos(theta_rad) * t
    dy = v_uav * np.sin(theta_rad) * t
    return uav_init + np.array([dx, dy, 0])

def trajectory_smoke_A(t, t_drop, t_delay, uav_init, v_uav, theta_uav, g=9.8, v_sink=3):
    t_det = t_drop + t_delay
    theta_rad = np.radians(theta_uav)
    if t < t_drop:
        return np.array([np.nan, np.nan, np.nan])
    elif t_drop <= t < t_det:
        t_free = t - t_drop
        x = uav_init[0] + v_uav * np.cos(theta_rad) * t
        y = uav_init[1] + v_uav * np.sin(theta_rad) * t
        z = uav_init[2] - 0.5 * g * t_free ** 2
        return np.array([x, y, z])
    else:
        t_sink = t - t_det
        x_det = uav_init[0] + v_uav * np.cos(theta_rad) * t_det
        y_det = uav_init[1] + v_uav * np.sin(theta_rad) * t_det
        z_det = uav_init[2] - 0.5 * g * t_delay ** 2
        z = z_det - v_sink * t_sink
        return np.array([x_det, y_det, z])

def generate_target_points_A(center=np.array([0, 200, 0]), r=7, h=10, n=5):
    pts = []
    for z in np.linspace(0, h, n):
        for rad in np.linspace(0, r, n):
            for theta in np.linspace(0, 2 * np.pi, n):
                x = center[0] + rad * np.cos(theta)
                y = center[1] + rad * np.sin(theta)
                pts.append(np.array([x, y, z]))
    return np.array(pts)

def is_shielded_A(t, missile_pos, smoke_pos, target_points, r_smoke=10):
    for tp in target_points:
        AB = tp - missile_pos
        AC = smoke_pos - missile_pos
        AB_sq = np.dot(AB, AB)
        if AB_sq < 1e-6:
            continue
        a = AB_sq
        b = -2 * np.dot(AC, AB)
        c = np.dot(AC, AC) - r_smoke ** 2
        disc = b ** 2 - 4 * a * c
        if disc < 0:
            continue
        s1 = (-b - np.sqrt(disc)) / (2 * a)
        s2 = (-b + np.sqrt(disc)) / (2 * a)
        if (0 <= s1 <= 1) or (0 <= s2 <= 1):
            return True
    return False

# ----------------- 数据初始化 -----------------
uav_inits = [
    np.array([17800, 0, 1800]),
    np.array([12000, 1400, 1400]),
    np.array([6000, -3000, 700]),
    np.array([11000, 2000, 1800]),
    np.array([13000, -2000, 1300])
]

missile_inits = [
    np.array([20000, 0, 2000]),
    np.array([19000, 600, 2100]),
    np.array([18000, -600, 1900])
]

target_fake = np.array([0, 0, 0])
t_missile_arrive = [np.linalg.norm(mi - target_fake) / 300 for mi in missile_inits]
target_points = generate_target_points_A()

def calculate_cost_matrix():
    cost_matrix = np.zeros((5, 3))
    for i in range(5):
        uav_init = uav_inits[i]
        for j in range(3):
            missile_init = missile_inits[j]
            min_dist = float('inf')
            for t in np.arange(0, t_missile_arrive[j], 0.5):
                uav_pos = trajectory_uav_A(t, uav_init, 120, 180)
                missile_pos = trajectory_missile_A(t, missile_init)
                dist = np.linalg.norm(uav_pos - missile_pos)
                if dist < min_dist:
                    min_dist = dist
            cost_matrix[i][j] = min_dist
    return cost_matrix

cost_matrix = calculate_cost_matrix()
row_ind, col_ind = linear_sum_assignment(cost_matrix)
assignment = {r: c for r, c in zip(row_ind, col_ind)}

# ----------------- 遗传算法个体结构 -----------------
# 基因结构: 前5个整数=各无人机烟幕弹数量(0~3), 后面按需拼接(每枚4参数)
MAX_PER_UAV = 3  # 上限
PARAMS_PER_SMOKE = 4  # theta,v,t_drop,t_delay

if "FitnessMax" not in creator.__dict__:
    creator.create("FitnessMax", base.Fitness, weights=(1.0, 1.0, 1.0))
if "Individual" not in creator.__dict__:
    creator.create("Individual", list, fitness=creator.FitnessMax)

def random_smoke_params():
    theta = random.uniform(0, 360)
    v = random.uniform(70, 140)
    t_drop = random.uniform(0, 60)
    t_delay = random.uniform(0, 10)
    return [theta, v, t_drop, t_delay]

def init_individual():
    ind = []
    # 数量基因
    for _ in range(5):
        ind.append(random.randint(0, MAX_PER_UAV))
    # 按数量追加参数
    for i in range(5):
        n = ind[i]
        for _ in range(n):
            ind.extend(random_smoke_params())
    return creator.Individual(ind)

def repair_individual(ind):
    # 约束前5个整数
    for i in range(5):
        ind[i] = int(ind[i])
        if ind[i] < 0:
            ind[i] = 0
        elif ind[i] > MAX_PER_UAV:
            ind[i] = MAX_PER_UAV
    needed = sum(ind[i] * PARAMS_PER_SMOKE for i in range(5))
    have = len(ind) - 5
    if have < needed:
        # 补足
        missing = needed - have
        blocks = missing // PARAMS_PER_SMOKE
        for _ in range(blocks):
            ind.extend(random_smoke_params())
    elif have > needed:
        # 截断
        del ind[5 + needed:]
    return ind

def evaluate(individual):
    repair_individual(individual)
    t_total = [0, 0, 0]
    intervals = [[], [], []]
    param_ptr = 5
    for i in range(5):
        n = individual[i]
        missile_idx = assignment.get(i)
        # 遍历该无人机每枚烟幕弹
        for _ in range(n):
            theta = individual[param_ptr]
            v = individual[param_ptr + 1]
            t_drop = individual[param_ptr + 2]
            t_delay = individual[param_ptr + 3]
            param_ptr += 4
            if missile_idx is None:
                continue
            uav_init = uav_inits[i]
            t_det = t_drop + t_delay
            t_end = min(t_det + 20, t_missile_arrive[missile_idx])
            if t_det >= t_end:
                continue
            valid_times = []
            for t in np.arange(t_det, t_end, 0.1):
                missile_pos = trajectory_missile_A(t, missile_inits[missile_idx])
                smoke_pos = trajectory_smoke_A(t, t_drop, t_delay, uav_init, v, theta)
                if np.any(np.isnan(smoke_pos)):
                    continue
                if is_shielded_A(t, missile_pos, smoke_pos, target_points):
                    valid_times.append(t)
            if valid_times:
                intervals[missile_idx].extend(np.unique(valid_times))
    for j in range(3):
        if intervals[j]:
            t_total[j] = len(np.unique(intervals[j])) * 0.1
    return tuple(t_total)

def mutate(ind):
    # 数量基因变异
    for i in range(5):
        if random.random() < 0.1:
            ind[i] = random.randint(0, MAX_PER_UAV)
    # 先修复以便参数索引有效
    repair_individual(ind)
    # 参数基因微扰
    for k in range(5, len(ind), 4):
        # theta
        if random.random() < 0.05:
            ind[k] = (ind[k] + random.uniform(-15, 15)) % 360
        # v
        if random.random() < 0.05:
            ind[k + 1] = min(140, max(70, ind[k + 1] + random.uniform(-5, 5)))
        # t_drop
        if random.random() < 0.05:
            ind[k + 2] = min(60, max(0, ind[k + 2] + random.uniform(-3, 3)))
        # t_delay
        if random.random() < 0.05:
            ind[k + 3] = min(10, max(0, ind[k + 3] + random.uniform(-1, 1)))
    # 再次修复(以防数量变异后不匹配)
    repair_individual(ind)
    return (ind,)

toolbox = base.Toolbox()
toolbox.register("individual", init_individual)
toolbox.register("population", tools.initRepeat, list, toolbox.individual, n=200)
toolbox.register("evaluate", evaluate)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", mutate)
toolbox.register("select", tools.selNSGA2)

pop = toolbox.population()
hof = tools.ParetoFront()
stats = tools.Statistics(lambda ind: ind.fitness.values)
stats.register("avg", np.mean)
stats.register("max", np.max)

pop, logbook = algorithms.eaSimple(
    pop, toolbox, cxpb=0.8, mutpb=0.2, ngen=300,
    stats=stats, halloffame=hof, verbose=True
)

best_ind = max(hof, key=lambda x: sum(x.fitness.values))
repair_individual(best_ind)

# ----------------- 结果输出 -----------------
result_data = []
param_ptr = 5
for i in range(5):
    n = best_ind[i]
    missile_idx = assignment.get(i)
    for k in range(n):
        theta = best_ind[param_ptr]
        v = best_ind[param_ptr + 1]
        t_drop = best_ind[param_ptr + 2]
        t_delay = best_ind[param_ptr + 3]
        param_ptr += 4
        if missile_idx is None:
            continue
        uav_id = f"FY{i + 1}"
        missile_id = f"M{missile_idx + 1}"
        uav_init = uav_inits[i]
        drop_pos = trajectory_uav_A(t_drop, uav_init, v, theta)
        det_pos = trajectory_smoke_A(t_drop + t_delay, t_drop, t_delay, uav_init, v, theta)
        t_det = t_drop + t_delay
        t_end = min(t_det + 20, t_missile_arrive[missile_idx])
        duration = 0.0
        for t in np.arange(t_det, t_end, 0.1):
            missile_pos = trajectory_missile_A(t, missile_inits[missile_idx])
            smoke_pos = trajectory_smoke_A(t, t_drop, t_delay, uav_init, v, theta)
            if np.any(np.isnan(smoke_pos)):
                continue
            if is_shielded_A(t, missile_pos, smoke_pos, target_points):
                duration += 0.1
        result_data.append([
            uav_id, theta, v, k + 1,
            drop_pos[0], drop_pos[1], drop_pos[2],
            det_pos[0], det_pos[1], det_pos[2],
            duration, missile_id
        ])

df = pd.DataFrame(result_data, columns=[
    "无人机编号", "运动方向(°)", "运动速度(m/s)", "烟幕弹编号",
    "投放点X(m)", "投放点Y(m)", "投放点Z(m)",
    "起爆点X(m)", "起爆点Y(m)", "起爆点Z(m)",
    "有效干扰时长(s)", "干扰导弹"
])
df.to_excel("result3.xlsx", index=False, engine="openpyxl")
print("《A题.pdf》小问5结果已写入result3.xlsx")