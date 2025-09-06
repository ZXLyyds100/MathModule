
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution
from core_functions import trajectory_missile_A, generate_target_points_A

# ---------------- 参数区 ----------------
DT = 0.1
POP_SIZE = 50
MAX_ITER = 100
USE_PARALLEL = False
PENALTY = 1e6
CHECK_SAMPLE = True

uav_init = np.array([17800, 0, 1800])
missile_init = np.array([20000, 0, 2000])
target_fake = np.array([0, 0, 0])
target_points = generate_target_points_A()
t_missile_arrive = np.linalg.norm(missile_init - target_fake) / 300.0
r_smoke = 10.0

print(f"导弹到达时间: {t_missile_arrive:.2f}s")
print(f"烟幕半径: {r_smoke}m")

# 预计算导弹轨迹
t_grid = np.arange(0, t_missile_arrive + DT, DT)
missile_traj = np.vstack([trajectory_missile_A(t, missile_init) for t in t_grid])
grid_dt = DT

def missile_pos_fast(t: float):
    if t <= 0:
        return missile_traj[0]
    if t >= t_grid[-1]:
        return missile_traj[-1]
    idx_f = t / grid_dt
    i0 = int(np.floor(idx_f))
    if i0 + 1 >= len(missile_traj):
        return missile_traj[-1]
    alpha = idx_f - i0
    return missile_traj[i0] * (1 - alpha) + missile_traj[i0 + 1] * alpha

def is_shielded_vec(missile_pos, smoke_pos, tp_batch, r_smoke=40.0):
    v = tp_batch - missile_pos
    v_norm2 = np.einsum('ij,ij->i', v, v)
    valid = v_norm2 > 1e-10
    if not np.any(valid):
        return False

    v_valid = v[valid]
    v_norm2_valid = v_norm2[valid]
    w = smoke_pos - missile_pos
    t_param = (v_valid @ w) / v_norm2_valid
    mask_t = (t_param >= 0.0) & (t_param <= 1.0)
    if not np.any(mask_t):
        return False

    v_sel = v_valid[mask_t]
    t_sel = t_param[mask_t][:, None]
    closest_pts = missile_pos + t_sel * v_sel
    diff = closest_pts - smoke_pos
    dist2 = np.einsum('ij,ij->i', diff, diff)
    return np.any(dist2 <= r_smoke * r_smoke)

def smoke_positions(times, t_drop, t_delay, v_uav, theta_deg,
                    g=9.8, v_sink=3.0):
    theta = np.radians(theta_deg)
    vx = v_uav * np.cos(theta)
    vy = v_uav * np.sin(theta)
    t_det = t_drop + t_delay
    pos = np.empty((len(times), 3), dtype=float)

    mask_pre = times < t_drop
    pos[mask_pre] = np.nan

    mask_free = (times >= t_drop) & (times < t_det)
    tfree = times[mask_free] - t_drop
    t_abs = times[mask_free]
    pos[mask_free, 0] = uav_init[0] + vx * t_abs
    pos[mask_free, 1] = uav_init[1] + vy * t_abs
    pos[mask_free, 2] = uav_init[2] - 0.5 * g * tfree * tfree

    mask_sink = times >= t_det
    tsink = times[mask_sink] - t_det
    x_det = uav_init[0] + vx * t_det
    y_det = uav_init[1] + vy * t_det
    z_det = uav_init[2] - 0.5 * g * (t_delay ** 2)
    pos[mask_sink, 0] = x_det
    pos[mask_sink, 1] = y_det
    pos[mask_sink, 2] = z_det - v_sink * tsink
    return pos

def fitness_core(theta, v, t_drop, t_delay):
    theta = theta % 360
    v = np.clip(v, 70, 140)
    t_drop = max(0.0, t_drop)
    t_delay = max(0.0, t_delay)
    t_det = t_drop + t_delay

    shield_duration = 20.0
    if t_det + shield_duration > t_missile_arrive:
        return PENALTY, 0.0

    t_start = t_det
    t_end = min(t_det + shield_duration, t_missile_arrive)
    if t_start >= t_end:
        return PENALTY, 0.0

    times = np.arange(t_start, t_end, DT)
    smoke_arr = smoke_positions(times, t_drop, t_delay, v, theta)
    effective = 0.0

    for sp, tt in zip(smoke_arr, times):
        if np.any(np.isnan(sp)):
            continue
        mp = missile_pos_fast(tt)
        if is_shielded_vec(mp, sp, target_points, r_smoke=r_smoke):
            effective += DT

    return -effective, effective

def find_valid_seeds(num_seeds=2000):
    """搜索有效参数种子"""
    print("搜索有效参数种子...")
    valid_seeds = []

    for i in range(num_seeds):
        theta = np.random.choice([
            np.random.uniform(0, 45),      # 东北方向
            np.random.uniform(315, 360),   # 西北方向
            np.random.uniform(270, 315),   # 西南方向
            np.random.uniform(45, 90)      # 东南方向
        ])
        v = np.random.uniform(80, 120)
        t_drop = np.random.uniform(0, 15)
        t_delay = np.random.uniform(0.5, 5)

        val, eff = fitness_core(theta, v, t_drop, t_delay)
        if eff > 0:
            valid_seeds.append((theta, v, t_drop, t_delay, eff))
            print(f"找到有效种子 {len(valid_seeds)}: θ={theta:.1f}°, eff={eff:.3f}s")

        if len(valid_seeds) >= 20:
            break

    return valid_seeds

def create_smart_population(valid_seeds, bounds, popsize):
    """改进的智能初始化 - 增加多样性"""
    np.random.seed(None)  # 使用真随机种子
    population = np.empty((popsize, 4))

    if valid_seeds:
        # 减少精英比例，增加随机性
        num_elite = min(int(popsize * 0.2), len(valid_seeds))  # 20%精英
        num_near = int(popsize * 0.4)  # 40%种子附近
        num_random = popsize - num_elite - num_near  # 40%完全随机

        valid_seeds_sorted = sorted(valid_seeds, key=lambda x: x[4], reverse=True)

        # 20%精英变异 - 增大变异幅度
        for i in range(num_elite):
            seed = valid_seeds_sorted[i % len(valid_seeds_sorted)]
            population[i, 0] = seed[0] + np.random.normal(0, 20)  # ±20°
            population[i, 1] = np.clip(seed[1] + np.random.normal(0, 10), 70, 140)  # ±10
            population[i, 2] = max(0, seed[2] + np.random.normal(0, 3))  # ±3s
            population[i, 3] = max(0, seed[3] + np.random.normal(0, 1.5))  # ±1.5s

        # 40%种子附近大范围搜索
        for i in range(num_elite, num_elite + num_near):
            seed_idx = np.random.randint(0, len(valid_seeds_sorted))
            seed = valid_seeds_sorted[seed_idx]
            population[i, 0] = seed[0] + np.random.normal(0, 45)  # ±45°
            population[i, 1] = np.random.uniform(70, 140)
            population[i, 2] = np.random.uniform(0, 10)
            population[i, 3] = np.random.uniform(0, 5)

        # 40%完全随机
        for i in range(num_elite + num_near, popsize):
            population[i, 0] = np.random.uniform(0, 360)
            population[i, 1] = np.random.uniform(70, 140)
            population[i, 2] = np.random.uniform(0, 20)
            population[i, 3] = np.random.uniform(0, 6)
    else:
        # 完全随机初始化
        for i in range(popsize):
            population[i, 0] = np.random.uniform(bounds[0][0], bounds[0][1])
            population[i, 1] = np.random.uniform(bounds[1][0], bounds[1][1])
            population[i, 2] = np.random.uniform(bounds[2][0], bounds[2][1])
            population[i, 3] = np.random.uniform(bounds[3][0], bounds[3][1])

    # 确保参数在边界内
    for i in range(4):
        population[:, i] = np.clip(population[:, i], bounds[i][0], bounds[i][1])

    print(f"改进种群生成: 精英{num_elite}个, 种子附近{num_near}个, 随机{num_random}个")
    return population

class OptimizationTracker:
    def __init__(self, popsize, maxiter):
        self.popsize = popsize
        self.maxiter = maxiter
        self.generation_bests = []
        self.current_gen_evaluations = []
        self.eval_count = 0
        self.global_best_eff = 0.0
        self.global_best_params = None

    def record_evaluation(self, params):
        self.eval_count += 1
        val, eff = fitness_core(*params)

        self.current_gen_evaluations.append(eff)

        if eff > self.global_best_eff:
            self.global_best_eff = eff
            self.global_best_params = params.copy()

        if len(self.current_gen_evaluations) >= self.popsize:
            gen_best = max(self.current_gen_evaluations)
            self.generation_bests.append(gen_best)

            current_gen = len(self.generation_bests)
            print(f"第{current_gen}代: 当代最优={gen_best:.4f}s, "
                  f"历史最优={self.global_best_eff:.4f}s")

            self.current_gen_evaluations = []

        return val

tracker = OptimizationTracker(POP_SIZE, MAX_ITER)

def objective_function(params):
    return tracker.record_evaluation(params)

def run_smart_de(valid_seeds):
    """改进的差分进化参数"""
    global tracker
    tracker = OptimizationTracker(POP_SIZE, MAX_ITER)

    bounds = [(0, 360), (70, 140), (0, 30), (0, 8)]
    init_population = create_smart_population(valid_seeds, bounds, POP_SIZE)

    result = differential_evolution(
        objective_function,
        bounds=bounds,
        maxiter=MAX_ITER,
        mutation=(0.5, 1.5),      # 增加变异幅度
        recombination=0.7,        # 降低交叉概率，保持多样性
        seed=None,                # 使用随机种子
        disp=True,
        polish=True,
        workers=1,
        init=init_population,
        atol=1e-6,               # 增加收敛容忍度
        tol=1e-6
    )

    if tracker.current_gen_evaluations:
        final_gen_best = max(tracker.current_gen_evaluations)
        tracker.generation_bests.append(final_gen_best)

    return result

def diagnose_fitness_landscape():
    """诊断适应度景观"""
    print("\n=== 适应度景观诊断 ===")

    # 测试几个已知好解附近的点
    test_params = [
        [353.4, 80.2, 0.7, 0.6],  # 最优种子
        [348.44, 95.41, 0.0, 0.01],  # DE结果
        [0, 100, 1, 1],  # 测试点1
        [180, 100, 5, 2],  # 测试点2
        [90, 100, 2, 1],   # 测试点3
        [270, 100, 3, 1.5], # 测试点4
    ]

    for i, params in enumerate(test_params):
        val, eff = fitness_core(*params)
        print(f"测试点{i+1}: θ={params[0]:.1f}°, v={params[1]:.1f}, "
              f"t_drop={params[2]:.1f}, t_delay={params[3]:.1f} -> eff={eff:.4f}s")

def plot_curve(best_eff):
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    gen_bests = tracker.generation_bests

    if len(gen_bests) == 0:
        print("警告: 无收敛数据")
        return

    non_zero_count = sum(1 for x in gen_bests if x > 0)
    print(f"非零代数: {non_zero_count}/{len(gen_bests)}")

    xs = np.arange(1, len(gen_bests) + 1)
    plt.figure(figsize=(14, 10))

    # 主图：收敛曲线
    plt.subplot(2, 2, 1)
    plt.plot(xs, gen_bests, 'b-o', linewidth=2, markersize=4, alpha=0.8)
    cumulative_best = np.maximum.accumulate(gen_bests)
    plt.plot(xs, cumulative_best, 'r-', linewidth=3, alpha=0.9)
    plt.xlabel('代数')
    plt.ylabel('遮蔽时长 (s)')
    plt.title(f'收敛过程 (半径{r_smoke}m)')
    plt.grid(True, alpha=0.3)
    plt.legend(['每代最优', '历史最优'])

    # 子图：改进检测
    plt.subplot(2, 2, 2)
    improvements = np.diff(cumulative_best)
    improved_gens = xs[1:][improvements > 0]
    if len(improved_gens) > 0:
        plt.scatter(improved_gens, improvements[improvements > 0],
                   color='orange', s=60, alpha=0.8, edgecolors='black')
        plt.xlabel('代数')
        plt.ylabel('改进幅度 (s)')
        plt.title(f'性能提升点 (共{len(improved_gens)}次)')
    else:
        plt.text(0.5, 0.5, '无显著改进', ha='center', va='center',
                transform=plt.gca().transAxes, fontsize=14)
        plt.title('改进历程')
    plt.grid(True, alpha=0.3)

    # 数据分布
    plt.subplot(2, 2, 3)
    unique_vals, counts = np.unique(gen_bests, return_counts=True)
    colors = ['red' if v == 0 else 'green' for v in unique_vals]
    plt.bar(range(len(unique_vals)), counts, color=colors, alpha=0.7)
    plt.xlabel('遮蔽时长值')
    plt.ylabel('出现次数')
    plt.title('数据分布')
    for i, (val, cnt) in enumerate(zip(unique_vals, counts)):
        plt.text(i, cnt + 0.1, f'{val:.2f}', ha='center', fontsize=9)

    # 统计信息
    plt.subplot(2, 2, 4)
    stats_text = f"""算法统计:
总代数: {len(gen_bests)}
有效代数: {non_zero_count}
最大遮蔽: {max(gen_bests):.4f}s
平均遮蔽: {np.mean(gen_bests):.4f}s
最终收敛: {gen_bests[-1]:.4f}s

参数设置:
种群规模: {POP_SIZE}
烟幕半径: {r_smoke}m
时间步长: {DT}s"""

    plt.text(0.1, 0.9, stats_text, transform=plt.gca().transAxes,
             fontsize=11, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    plt.axis('off')

    plt.tight_layout()
    plt.savefig('小问2_智能优化收敛曲线.png', dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()

def main():
    print("=== 智能差分进化优化 ===")

    # 诊断适应度景观
    diagnose_fitness_landscape()

    # 第一步：搜索有效种子
    valid_seeds = find_valid_seeds(2000)

    if valid_seeds:
        print(f"\n成功找到 {len(valid_seeds)} 个有效种子")
        print("最优种子:")
        valid_seeds.sort(key=lambda x: x[4], reverse=True)
        for i, seed in enumerate(valid_seeds[:5]):
            print(f"  {i+1}. θ={seed[0]:.1f}°, v={seed[1]:.1f}, "
                  f"t_drop={seed[2]:.1f}, t_delay={seed[3]:.1f}, eff={seed[4]:.3f}s")
    else:
        print("警告: 未找到有效种子，使用随机初始化")

    # 第二步：基于种子的智能优化
    print(f"\n开始智能差分进化(种群{POP_SIZE}, 迭代{MAX_ITER})...")
    result = run_smart_de(valid_seeds)

    # 第三步：结果分析
    best_params = result.x
    best_val, best_eff = fitness_core(*best_params)

    print(f"\n=== 最终优化结果 ===")
    print(f'DE最优参数: θ={best_params[0]:.2f}°, v={best_params[1]:.2f}m/s')
    print(f'投放策略: t_drop={best_params[2]:.2f}s, t_delay={best_params[3]:.2f}s')
    print(f'DE最优遮蔽: {best_eff:.4f}s')

    if tracker.global_best_eff > best_eff:
        print(f'全局最优遮蔽: {tracker.global_best_eff:.4f}s (更优)')
        print(f'全局最优参数: θ={tracker.global_best_params[0]:.2f}°, '
              f'v={tracker.global_best_params[1]:.2f}, '
              f't_drop={tracker.global_best_params[2]:.2f}, '
              f't_delay={tracker.global_best_params[3]:.2f}')

    print(f'评估次数: {tracker.eval_count}, 记录代数: {len(tracker.generation_bests)}')

    plot_curve(max(best_eff, tracker.global_best_eff))

if __name__ == '__main__':
    main()