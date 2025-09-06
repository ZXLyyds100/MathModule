import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution
from core_functions import trajectory_missile_A, generate_target_points_A, trajectory_uav_A

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# ---------------- 参数区 ----------------
DT = 0.1
POP_SIZE = 80  # 增加种群规模
MAX_ITER = 150  # 增加迭代次数
USE_PARALLEL = False
PENALTY = 1e6
CHECK_SAMPLE = True

uav_init = np.array([17800, 0, 1800])
missile_init = np.array([20000, 0, 2000])
target_fake = np.array([0, 0, 0])
target_points = generate_target_points_A()
t_missile_arrive = np.linalg.norm(missile_init - target_fake) / 300.0
r_smoke = 10.0
shield_duration = 20.0

print(f"导弹到达时间: {t_missile_arrive:.2f}s")
print(f"烟幕半径: {r_smoke}m")
print(f"烟幕时长: {shield_duration}s")

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


def is_shielded_vec(missile_pos, smoke_pos, tp_batch, r_smoke=10.0):
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


def smoke_positions(times, t_drop, t_delay, v_uav, theta_deg, g=9.8, v_sink=3.0):
    times = np.atleast_1d(times)
    theta = np.radians(theta_deg)
    vx = v_uav * np.cos(theta)
    vy = v_uav * np.sin(theta)
    t_det = t_drop + t_delay
    pos = np.empty((len(times), 3), dtype=float)

    mask_pre = times < t_drop
    pos[mask_pre] = np.nan

    mask_free = (times >= t_drop) & (times < t_det)
    if np.any(mask_free):
        tfree = times[mask_free] - t_drop
        t_abs = times[mask_free]
        pos[mask_free, 0] = uav_init[0] + vx * t_abs
        pos[mask_free, 1] = uav_init[1] + vy * t_abs
        pos[mask_free, 2] = uav_init[2] - 0.5 * g * tfree * tfree

    mask_sink = times >= t_det
    if np.any(mask_sink):
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


def find_valid_seeds(num_seeds=8000):  # 大幅增加搜索数量
    """针对小半径问题的强化种子搜索"""
    print("执行强化种子搜索（针对10m半径优化）...")
    valid_seeds = []

    # 多策略角度采样
    angle_strategies = [
        # 策略1：8个主要方向密集搜索
        lambda: np.random.choice([0, 45, 90, 135, 180, 225, 270, 315]) + np.random.normal(0, 15),
        # 策略2：假目标方向周围
        lambda: 180 + np.random.normal(0, 30),
        # 策略3：垂直于导弹-假目标线
        lambda: np.random.choice([90, 270]) + np.random.normal(0, 20),
        # 策略4：完全随机
        lambda: np.random.uniform(0, 360)
    ]

    for i in range(num_seeds):
        # 选择角度策略
        if i < num_seeds * 0.4:  # 40%主要方向
            theta = angle_strategies[0]()
        elif i < num_seeds * 0.6:  # 20%假目标方向
            theta = angle_strategies[1]()
        elif i < num_seeds * 0.8:  # 20%垂直方向
            theta = angle_strategies[2]()
        else:  # 20%随机
            theta = angle_strategies[3]()

        # 速度策略：偏向高速（提高机动性）
        if np.random.random() < 0.7:
            v = np.random.uniform(100, 140)  # 70%选择高速
        else:
            v = np.random.uniform(70, 100)  # 30%选择中低速

        # 时间策略：早投放，短延迟
        t_drop = np.random.exponential(3)  # 指数分布，偏向早期
        t_drop = min(t_drop, 15)

        t_delay = np.random.exponential(1.5)  # 指数分布，偏向短延迟
        t_delay = min(max(t_delay, 0.2), 5)

        val, eff = fitness_core(theta, v, t_drop, t_delay)
        if eff > 0:
            valid_seeds.append((theta, v, t_drop, t_delay, eff))
            print(
                f"种子{len(valid_seeds)}: θ={theta:.1f}°, v={v:.1f}, t_drop={t_drop:.1f}, t_delay={t_delay:.1f}, eff={eff:.3f}s")

        if len(valid_seeds) >= 50:  # 增加目标种子数
            break

    print(f"强化搜索完成，找到{len(valid_seeds)}个有效种子")
    return valid_seeds


def create_smart_population(valid_seeds, bounds, popsize):
    """多层次智能初始化"""
    np.random.seed(None)
    population = np.empty((popsize, 4))

    if valid_seeds:
        valid_seeds_sorted = sorted(valid_seeds, key=lambda x: x[4], reverse=True)

        # 精英保护：30%
        num_elite = min(int(popsize * 0.3), len(valid_seeds))
        # 种子扰动：40%
        num_perturb = int(popsize * 0.4)
        # 完全随机：30%
        num_random = popsize - num_elite - num_perturb

        # 30%精英个体（小幅扰动）
        for i in range(num_elite):
            seed = valid_seeds_sorted[i % len(valid_seeds)]
            population[i, 0] = seed[0] + np.random.normal(0, 8)  # ±8°
            population[i, 1] = np.clip(seed[1] + np.random.normal(0, 3), 70, 140)
            population[i, 2] = max(0, seed[2] + np.random.normal(0, 0.8))
            population[i, 3] = max(0.1, seed[3] + np.random.normal(0, 0.3))

        # 40%种子大幅扰动
        for i in range(num_elite, num_elite + num_perturb):
            seed_idx = np.random.randint(0, len(valid_seeds))
            seed = valid_seeds_sorted[seed_idx]
            population[i, 0] = seed[0] + np.random.normal(0, 35)  # ±35°
            population[i, 1] = np.random.uniform(80, 140)  # 高速偏好
            population[i, 2] = np.random.uniform(0, 8)  # 早投放
            population[i, 3] = np.random.uniform(0.2, 4)  # 短延迟

        # 30%完全随机（保持多样性）
        for i in range(num_elite + num_perturb, popsize):
            population[i, 0] = np.random.uniform(0, 360)
            population[i, 1] = np.random.uniform(90, 140)  # 偏向高速
            population[i, 2] = np.random.uniform(0, 6)
            population[i, 3] = np.random.uniform(0.3, 3)
    else:
        # 无种子时的智能随机
        for i in range(popsize):
            population[i, 0] = np.random.uniform(0, 360)
            population[i, 1] = np.random.uniform(90, 140)
            population[i, 2] = np.random.uniform(0, 5)
            population[i, 3] = np.random.uniform(0.5, 2.5)

    # 边界约束
    for i in range(4):
        population[:, i] = np.clip(population[:, i], bounds[i][0], bounds[i][1])

    print(f"智能种群：{num_elite}精英 + {num_perturb}扰动 + {num_random}随机")
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
            print(f"第{current_gen:3d}代: 当代最优={gen_best:.4f}s, 历史最优={self.global_best_eff:.4f}s")

            self.current_gen_evaluations = []

        return val


tracker = OptimizationTracker(POP_SIZE, MAX_ITER)


def objective_function(params):
    return tracker.record_evaluation(params)


def run_smart_de(valid_seeds):
    """针对困难问题的DE参数调优"""
    global tracker
    tracker = OptimizationTracker(POP_SIZE, MAX_ITER)

    bounds = [(0, 360), (70, 140), (0, 20), (0, 8)]
    init_population = create_smart_population(valid_seeds, bounds, POP_SIZE)

    result = differential_evolution(
        objective_function,
        bounds=bounds,
        maxiter=MAX_ITER,
        mutation=(0.4, 1.0),  # 适中变异
        recombination=0.8,  # 高交叉率
        strategy='best1bin',  # 最优策略
        seed=None,
        disp=True,
        polish=True,
        workers=1,
        init=init_population,
        atol=1e-8,
        tol=1e-8,
        updating='deferred'
    )

    if tracker.current_gen_evaluations:
        final_gen_best = max(tracker.current_gen_evaluations)
        tracker.generation_bests.append(final_gen_best)

    return result


def diagnose_fitness_landscape():
    print("\n=== 适应度景观诊断 ===")
    test_params = [
        [0, 120, 1, 1],  # 正东高速
        [90, 120, 1, 1],  # 正北高速
        [180, 120, 1, 1],  # 正西高速
        [270, 120, 1, 1],  # 正南高速
        [45, 110, 2, 1.5],  # 东北中速
        [225, 110, 2, 1.5],  # 西南中速
        [315, 100, 0.5, 0.8],  # 西北早投
        [135, 100, 0.5, 0.8],  # 东南早投
    ]

    for i, params in enumerate(test_params):
        val, eff = fitness_core(*params)
        print(
            f"测试{i + 1}: θ={params[0]:3.0f}° v={params[1]:3.0f} t_drop={params[2]:.1f} t_delay={params[3]:.1f} -> {eff:.4f}s")


def plot_curve(best_eff):
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    gen_bests = tracker.generation_bests

    if len(gen_bests) == 0:
        print("警告: 无收敛数据")
        return

    non_zero_count = sum(1 for x in gen_bests if x > 0)
    print(f"有效代数: {non_zero_count}/{len(gen_bests)}")

    xs = np.arange(1, len(gen_bests) + 1)
    plt.figure(figsize=(14, 10))

    # 主收敛曲线
    plt.subplot(2, 2, 1)
    plt.plot(xs, gen_bests, 'b-o', linewidth=2, markersize=3, alpha=0.8)
    cumulative_best = np.maximum.accumulate(gen_bests)
    plt.plot(xs, cumulative_best, 'r-', linewidth=3, alpha=0.9)
    plt.xlabel('代数')
    plt.ylabel('遮蔽时长 (s)')
    plt.title(f'强化DE收敛过程 (半径{r_smoke}m, 时长{shield_duration}s)')
    plt.grid(True, alpha=0.3)
    plt.legend(['每代最优', '历史最优'])

    # 改进检测
    plt.subplot(2, 2, 2)
    improvements = np.diff(cumulative_best)
    improved_gens = xs[1:][improvements > 1e-6]
    if len(improved_gens) > 0:
        plt.scatter(improved_gens, improvements[improvements > 1e-6],
                    color='orange', s=50, alpha=0.8, edgecolors='black')
        plt.xlabel('代数')
        plt.ylabel('改进幅度 (s)')
        plt.title(f'性能提升记录 (共{len(improved_gens)}次)')
    else:
        plt.text(0.5, 0.5, '无显著改进', ha='center', va='center',
                 transform=plt.gca().transAxes, fontsize=12)
        plt.title('改进历程')
    plt.grid(True, alpha=0.3)

    # 数据分布
    plt.subplot(2, 2, 3)
    # 按0.1s间隔分组
    bins = np.arange(0, max(gen_bests) + 0.15, 0.1)
    counts, bin_edges = np.histogram(gen_bests, bins=bins)
    colors = ['red' if edge < 0.05 else 'green' for edge in bin_edges[:-1]]

    plt.bar(range(len(counts)), counts, color=colors, alpha=0.7, edgecolor='black')
    plt.xlabel('遮蔽时长区间')
    plt.ylabel('出现次数')
    plt.title('结果分布直方图')

    # 设置x轴标签
    tick_labels = [f'{edge:.1f}' for edge in bin_edges[:-1]]
    plt.xticks(range(len(counts)), tick_labels, rotation=45)

    # 统计信息
    plt.subplot(2, 2, 4)
    avg_eff = np.mean(gen_bests)
    max_eff = max(gen_bests)
    zero_ratio = sum(1 for x in gen_bests if x < 0.001) / len(gen_bests)

    stats_text = f"""强化DE算法统计:
总代数: {len(gen_bests)}
有效代数: {non_zero_count}
无效比例: {zero_ratio:.1%}

性能指标:
最大遮蔽: {max_eff:.4f}s
平均遮蔽: {avg_eff:.4f}s
最终收敛: {gen_bests[-1]:.4f}s

参数设置:
种群规模: {POP_SIZE}
烟幕半径: {r_smoke}m
烟幕时长: {shield_duration}s
时间步长: {DT}s"""

    plt.text(0.05, 0.95, stats_text, transform=plt.gca().transAxes,
             fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    plt.axis('off')

    plt.tight_layout()
    plt.savefig('小问2_强化优化收敛曲线.png', dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()


def main():
    print("=== 小问2强化差分进化优化 ===")

    # 诊断适应度景观
    diagnose_fitness_landscape()

    # 强化种子搜索
    valid_seeds = find_valid_seeds(8000)

    if valid_seeds:
        print(f"\n成功找到 {len(valid_seeds)} 个有效种子")
        print("最优种子:")
        valid_seeds.sort(key=lambda x: x[4], reverse=True)
        for i, seed in enumerate(valid_seeds[:8]):
            print(f"  {i + 1}. θ={seed[0]:.1f}°, v={seed[1]:.1f}, "
                  f"t_drop={seed[2]:.1f}, t_delay={seed[3]:.1f}, eff={seed[4]:.3f}s")
    else:
        print("警告: 未找到有效种子，问题可能过于困难")
        return

    # 强化差分进化
    print(f"\n开始强化差分进化(种群{POP_SIZE}, 迭代{MAX_ITER})...")
    result = run_smart_de(valid_seeds)

    # 结果分析
    best_params = result.x
    best_val, best_eff = fitness_core(*best_params)

    print(f"\n=== 强化优化最终结果 ===")
    print(f'DE最优参数: θ={best_params[0]:.2f}°, v={best_params[1]:.2f}m/s')
    print(f'投放策略: t_drop={best_params[2]:.2f}s, t_delay={best_params[3]:.2f}s')
    print(f'DE最优遮蔽: {best_eff:.4f}s')

    if tracker.global_best_eff > best_eff:
        print(f'全局最优遮蔽: {tracker.global_best_eff:.4f}s (历史最佳)')
        print(f'全局最优参数: θ={tracker.global_best_params[0]:.2f}°, '
              f'v={tracker.global_best_params[1]:.2f}, '
              f't_drop={tracker.global_best_params[2]:.2f}, '
              f't_delay={tracker.global_best_params[3]:.2f}')
        final_params = tracker.global_best_params
        final_eff = tracker.global_best_eff
    else:
        final_params = best_params
        final_eff = best_eff

    print(f'总评估次数: {tracker.eval_count}')

    # 输出关键位置
    theta_opt, v_opt, t_drop_opt, t_delay_opt = final_params
    drop_pos = trajectory_uav_A(t_drop_opt, uav_init, v_opt, theta_opt)
    t_det_opt = t_drop_opt + t_delay_opt
    det_pos = smoke_positions(np.array([t_det_opt]), t_drop_opt, t_delay_opt, v_opt, theta_opt)[0]

    print(f'\n投放点坐标: ({drop_pos[0]:.2f}, {drop_pos[1]:.2f}, {drop_pos[2]:.2f}) m')
    print(f'起爆点坐标: ({det_pos[0]:.2f}, {det_pos[1]:.2f}, {det_pos[2]:.2f}) m')

    plot_curve(final_eff)


if __name__ == '__main__':
    main()