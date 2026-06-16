"""
三轴 / 无侧限试验数据一键处理脚本
============================================================
功能：
  1. 搜索最深子文件夹，提取数据文件（.slidata / .csv）
  2. 提取第23列（轴向应变）和第29列（偏应力）
  3. 按 15% 应变规则计算峰值偏应力
  4. 平行试样分组，剔除异常值，计算均值、SD、CV
============================================================
输出：
  一键生成平均值/
    ├── 峰值/峰值汇总.csv
    └── 平均值/平行试样结果.csv
============================================================
"""

import csv
import os
import subprocess
import sys

import numpy as np

# ╔══════════════════════════════════════════════════════════╗
# ║  ★★★  请替换下面这行路径为你实际的源数据文件夹  ★★★  ║
# ╚══════════════════════════════════════════════════════════╝
SOURCE_FOLDER = r"F:\1"
# 说明：该文件夹是试验仪器导出的根目录，内含多层子文件夹，
#       每个最深子文件夹里有一个试验数据文件(.slidata 或 .csv)
# ╔══════════════════════════════════════════════════════════╗
# ║  ★★★  请替换上面这行路径为你实际的源数据文件夹  ★★★  ║
# ╚══════════════════════════════════════════════════════════╝

# 输出根目录（无需修改）
BASE_OUTPUT = r"F:\数据"

# 15% 应变阈值（三轴试验用；无侧限试验峰值通常在 1~3%，不受影响）
STRAIN_THRESHOLD = 15.0

# 平行试样异常值剔除阈值（偏离均值超过此值则剔除）
OUTLIER_THRESHOLD = 200.0

# ============================================================


def ask_yes_no(prompt: str, default: bool = False) -> bool:
    """询问是否执行某一步，直接回车时使用默认值。"""
    default_text = "Y/n" if default else "y/N"
    while True:
        answer = input(f"{prompt} ({default_text}): ").strip().lower()
        if not answer:
            return default
        if answer in {"y", "yes", "是", "需要", "1"}:
            return True
        if answer in {"n", "no", "否", "不", "不需要", "0"}:
            return False
        print("  请输入 y 或 n。")


def find_data_files(source_folder: str) -> list:
    """搜索最深子文件夹中的第一个数据文件"""
    deepest_folders = []

    for root, dirs, files in os.walk(source_folder):
        if not dirs:
            deepest_folders.append(root)

    if not deepest_folders:
        deepest_folders = [source_folder]

    data_files = []
    for folder in deepest_folders:
        try:
            all_files = os.listdir(folder)
        except PermissionError:
            continue

        sli_files = sorted(f for f in all_files if f.endswith(".slidata"))
        csv_files = sorted(f for f in all_files if f.endswith(".csv"))

        if sli_files:
            target = sli_files[0]
        elif csv_files:
            target = csv_files[0]
        elif all_files:
            target = all_files[0]
        else:
            continue

        data_files.append(os.path.join(folder, target))

    return data_files


def extract_strain_stress(file_path: str) -> tuple:
    """
    从数据文件中提取第23列(轴向应变)和第29列(偏应力)
    返回 (样本名, strain_array, stress_array)
    """
    sample_name = os.path.splitext(os.path.basename(file_path))[0]
    strain, stress = [], []

    with open(file_path, "r", encoding="utf-8", errors="replace") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) >= 29:
                try:
                    s23 = float(row[22])   # 第23列：轴向应变 (%)
                    s29 = float(row[28])   # 第29列：偏应力 (kPa)
                    strain.append(s23)
                    stress.append(s29)
                except (ValueError, IndexError):
                    continue

    return sample_name, np.array(strain), np.array(stress)


def find_peak(strain: np.ndarray, stress: np.ndarray) -> tuple:
    """
    15% 应变规则计算峰值
    若峰值对应应变 > 15%，取 15% 应变处的应力；否则取实际最大值
    返回 (peak_stress, strain_at_peak)
    """
    if len(stress) == 0:
        return 0.0, 0.0

    max_idx = int(np.argmax(stress))
    max_stress = float(stress[max_idx])
    strain_at_max = float(strain[max_idx])

    if strain_at_max > STRAIN_THRESHOLD:
        idx_15 = int(np.argmin(np.abs(strain - STRAIN_THRESHOLD)))
        return float(stress[idx_15]), float(strain[idx_15])

    return max_stress, strain_at_max


def group_samples(results: list) -> dict:
    """
    按试样名称分组，取最后一个 '-' 之前的部分作为组名
    例如 'A-1' -> 'A', 'B-2' -> 'B'
    """
    groups = {}
    for sample_name, peak_stress, _ in results:
        group_name = sample_name.rsplit("-", 1)[0]
        groups.setdefault(group_name, []).append(peak_stress)
    return groups


def plot_stress_strain(plot_data: list, output_folder: str) -> tuple:
    """
    参照“6.应力应变图.py”绘制应力应变散点图。
    返回 (成功数量, 失败数量)。
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("  [错误] 未安装 matplotlib，无法绘制应力应变图。")
        print("  可先安装 matplotlib，或选择不绘图后重新运行。")
        return 0, len(plot_data)

    plt.rcParams["font.sans-serif"] = ["SimHei"]
    plt.rcParams["axes.unicode_minus"] = False

    os.makedirs(output_folder, exist_ok=True)
    success = 0
    failed = 0

    for sample_name, strain, stress in plot_data:
        try:
            plt.figure(figsize=(10, 6))
            plt.scatter(strain, stress, color="blue", s=10, label="Data Points")
            plt.title(f"Scatter Plot for {sample_name}")
            plt.xlabel("Axial Strain (%)")
            plt.ylabel("Deviator Stress (kPa)")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()

            output_image_path = os.path.join(output_folder, f"{sample_name}.png")
            plt.savefig(output_image_path, dpi=300)
            plt.close()

            print(f"  已生成: {output_image_path}")
            success += 1
        except Exception as e:
            plt.close()
            print(f"  [失败] {sample_name}: {e}")
            failed += 1

    return success, failed


def main() -> None:
    print("=" * 55)
    print("  三轴 / 无侧限试验数据一键处理")
    print("=" * 55)

    # ---- 校验源路径 ----
    if not os.path.exists(SOURCE_FOLDER):
        print(f"\n  [错误] 源文件夹不存在: {SOURCE_FOLDER}")
        print("  请修改脚本顶部的 SOURCE_FOLDER 路径后重试。")
        input("\n  按任意键退出...")
        sys.exit(1)

    # ---- 创建输出目录 ----
    peak_dir = os.path.join(BASE_OUTPUT, "峰值")
    avg_dir = os.path.join(BASE_OUTPUT, "平均值")
    plot_dir = os.path.join(BASE_OUTPUT, "应力应变图")
    os.makedirs(peak_dir, exist_ok=True)
    os.makedirs(avg_dir, exist_ok=True)

    draw_plots = ask_yes_no("\n是否需要绘制应力应变图？", default=False)
    total_steps = 5 if draw_plots else 4

    # ---- 1. 搜索数据文件 ----
    print(f"\n[1/{total_steps}] 搜索最深子文件夹中的数据文件...")
    data_files = find_data_files(SOURCE_FOLDER)
    print(f"       找到 {len(data_files)} 个文件")
    if not data_files:
        print("  [错误] 未找到任何数据文件 (.slidata / .csv)")
        input("\n  按任意键退出...")
        sys.exit(1)

    # ---- 2. 提取应变-应力 & 计算峰值 ----
    print(f"\n[2/{total_steps}] 提取第23列(轴向应变)和第29列(偏应力)，计算峰值...")
    results = []  # [(sample_name, peak_stress, strain_at_peak), ...]
    plot_data = []  # [(sample_name, strain_array, stress_array), ...]
    skipped = 0

    for fp in data_files:
        name, strain, stress = extract_strain_stress(fp)
        if len(stress) == 0:
            print(f"  [跳过] {name} — 无有效数据")
            skipped += 1
            continue
        plot_data.append((name, strain, stress))
        peak, strain_at = find_peak(strain, stress)
        results.append((name, peak, strain_at))
        print(f"  {name}: 峰值 = {peak:.2f} kPa  @ 应变 = {strain_at:.2f}%")

    if not results:
        print("\n  [错误] 所有文件均无有效数据，请检查数据格式")
        input("\n  按任意键退出...")
        sys.exit(1)

    # ---- 3. 保存峰值汇总 ----
    print(f"\n[3/{total_steps}] 保存峰值汇总...")
    peak_csv = os.path.join(peak_dir, "峰值汇总.csv")
    with open(peak_csv, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["试样名称", "峰值偏应力(kPa)", "对应应变(%)"])
        for name, peak, strain_at in results:
            w.writerow([name, round(peak, 3), round(strain_at, 3)])
    print(f"       已保存: {peak_csv}")

    # ---- 4. 平行试样平均值 / SD / CV ----
    print(f"\n[4/{total_steps}] 计算平行试样平均值、SD、CV（剔除异常值）...")
    groups = group_samples(results)

    avg_csv = os.path.join(avg_dir, "平行试样结果.csv")
    with open(avg_csv, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["试样组名", "试样数", "平均峰值(kPa)", "标准差SD(kPa)", "变异系数CV(%)"])

        for group_name in sorted(groups.keys()):
            values = np.array(groups[group_name], dtype=float)
            n = len(values)

            if n < 2:
                avg = float(values[0])
                sd = 0.0
                cv = 0.0
            else:
                # 第一轮平均
                avg_initial = float(np.mean(values))
                # 剔除偏离均值超过阈值的异常值
                valid = np.array([v for v in values if abs(v - avg_initial) <= OUTLIER_THRESHOLD])
                n_valid = len(valid)

                if n_valid >= 1:
                    avg = float(np.mean(valid))
                    sd = float(np.std(valid, ddof=1)) if n_valid >= 2 else 0.0
                else:
                    avg = avg_initial
                    sd = float(np.std(values, ddof=1))

                cv = (sd / avg * 100.0) if avg != 0 else 0.0

            w.writerow([group_name, n, round(avg, 3), round(sd, 3), round(cv, 3)])
            print(f"  {group_name}: n={n}, 均值={avg:.2f} kPa, SD={sd:.2f}, CV={cv:.2f}%")

    print(f"\n       已保存: {avg_csv}")

    plot_success = 0
    plot_failed = 0
    if draw_plots:
        print(f"\n[5/{total_steps}] 绘制应力应变图...")
        plot_success, plot_failed = plot_stress_strain(plot_data, plot_dir)
        print(f"       已完成: 成功 {plot_success} 张，失败 {plot_failed} 张")

    # ---- 完成 ----
    print("\n" + "=" * 55)
    print(f"  处理完成！（有效 {len(results)} 个，跳过 {skipped} 个）")
    print(f"  峰值汇总 → {peak_csv}")
    print(f"  平行试样 → {avg_csv}")
    if draw_plots:
        print(f"  应力应变图 → {plot_dir}（成功 {plot_success} 张，失败 {plot_failed} 张）")
    print("=" * 55)

    # 自动打开输出文件夹
    subprocess.Popen(["explorer", BASE_OUTPUT])
    input("\n  按任意键退出...")


if __name__ == "__main__":
    main()
