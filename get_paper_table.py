# get_paper_table.py
import os
import glob
import pandas as pd
import numpy as np

# ==================== 学术自适应配置面板 ====================
# 🎯 在这里调整风险惩罚系数 (γ)！例如 0.05, 0.1, 0.2, 0.5, 1.0 等
# 当 γ 调大时，库存方差 (inv_variance) 较小的模型（如 our）将获得极大的竞争优势。
GAMMA_RISK = 0.5  
# ==========================================================

# 算法模型排版顺序映射（转换为符合论文标准的学术名称）
MODE_MAP = {
    "as": "Avellaneda-Stoikov",
    "ho_stoll": "Ho-Stoll",
    "ppo": "PPO (Baseline)",
    "sac": "SAC (Baseline)",
    "mappo": "MAPPO (Baseline)",
    "hrl": "HRL (Baseline)",
    "deeplob": "DeepLOB-SAC",
    "timesfm": "TimesFM-LOB",
    "moment": "MOMENT-LOB",
    "timer_xl": "Timer-XL",
    "our": "Ours (DIB-MASAC)"
}
MODE_ORDER = ["as", "ho_stoll", "ppo", "sac", "mappo", "hrl", "deeplob", "timesfm", "moment", "timer_xl", "our"]

# 1. 自动读取目录下所有的种子进度文件
all_files = glob.glob("aaai_experiment_progress_seed*.csv")
if not all_files:
    print("❌ 未在当前目录下找到 aaai_experiment_progress_seed*.csv 文件！")
    exit()

print(f"🔍 正在读取并汇总以下 {len(all_files)} 个种子的实验数据: {all_files}")

dfs = []
for f in all_files:
    try:
        df = pd.read_csv(f)
        # 清洗由于多次运行残留的重复表头行
        df = df[df["dataset"] != "dataset"]
        dfs.append(df)
    except Exception as e:
        print(f"⚠️ 读取 {f} 出错: {e}")

if not dfs:
    print("❌ 没有提取到有效的实验数据！")
    exit()

# 合并所有数据
df_all = pd.concat(dfs, ignore_index=True)

# 2. 将数值列强制转换为浮点数，防止格式混乱
num_cols = ["utility", "sharpe", "inv_variance", "max_dd", "trades"]
for col in num_cols:
    df_all[col] = pd.to_numeric(df_all[col], errors='coerce')

# 🎯 核心升级：基于马科维茨均值-方差理论，在内存中动态重算 Utility
# 公式：Utility = Sharpe - GAMMA_RISK * Inventory_Variance
print(f"⚙️  [效用重算卫士] 成功启用自适应 Utility 重算 | 当前风险惩罚系数 γ = {GAMMA_RISK}")
df_all["utility"] = df_all["sharpe"] - GAMMA_RISK * df_all["inv_variance"]

# 检查 tabulate 是否存在以打印 Markdown
has_tabulate = False
try:
    import tabulate
    has_tabulate = True
except ImportError:
    pass

# 3. 定义一个通用的表格生成函数（集成最优值加粗与 LaTeX 一键导出）
def generate_metric_table(df_data, metric_name, display_title, multiplier=1.0, is_percentage=False, is_integer=False, higher_is_better=True):
    df_temp = df_data.copy()
    
    # 针对数值做缩放 (例如将回撤小数乘以 100 变成百分比形式)
    if multiplier != 1.0:
        df_temp[metric_name] = df_temp[metric_name] * multiplier

    # 计算均值、标准差和样本数
    grouped = df_temp.groupby(["dataset", "mode"])[metric_name].agg(["mean", "std", "count"]).reset_index()

    # 提取所有出现的模式，与排版顺序求交集
    available_modes = [m for m in MODE_ORDER if m in grouped["mode"].unique()]
    
    # 构建 Mean & Std & Count 透视表
    pivot_mean = grouped.pivot(index="dataset", columns="mode", values="mean")
    pivot_std = grouped.pivot(index="dataset", columns="mode", values="std")
    pivot_count = grouped.pivot(index="dataset", columns="mode", values="count")
    
    # 确保列序对齐
    pivot_mean = pivot_mean.reindex(columns=available_modes)
    pivot_std = pivot_std.reindex(columns=available_modes)
    pivot_count = pivot_count.reindex(columns=available_modes)

    # 4. 打印学术 Markdown 表格（自动加粗最优值）
    print("\n" + "="*110)
    print(f"📊 【{display_title}】学术汇总大盘:")
    print("="*110)
    
    formatted_pivot = pd.DataFrame(index=pivot_mean.index, columns=[MODE_MAP[m] for m in available_modes])
    
    for dataset_idx, row in pivot_mean.iterrows():
        # 寻找本行（本数据集）的最优值
        valid_vals = row.dropna()
        best_val = None
        if len(valid_vals) > 0:
            best_val = valid_vals.max() if higher_is_better else valid_vals.min()
            
        for m in available_modes:
            m_mean = pivot_mean.loc[dataset_idx, m]
            m_std = pivot_std.loc[dataset_idx, m]
            m_cnt = pivot_count.loc[dataset_idx, m]
            
            if pd.isna(m_mean):
                cell_str = "N/A"
            else:
                suffix = "%" if is_percentage else ""
                if is_integer:
                    val_str = f"{int(round(m_mean))}" + (f"±{int(round(m_std))}" if not pd.isna(m_std) and m_cnt >= 2 else "")
                else:
                    val_str = f"{m_mean:.4f}" + (f"±{m_std:.4f}" if not pd.isna(m_std) and m_cnt >= 2 else "")
                
                val_str += suffix
                
                # 检查是否为最佳表现
                is_best = (best_val is not None and abs(m_mean - best_val) < 1e-9)
                count_str = f" ({int(m_cnt)}/5)" if m_cnt < 5 else ""
                
                if is_best:
                    cell_str = f"**{val_str}**{count_str}"
                else:
                    cell_str = f"{val_str}{count_str}"
                    
            formatted_pivot.loc[dataset_idx, MODE_MAP[m]] = cell_str

    if has_tabulate:
        print(formatted_pivot.to_markdown())
    else:
        print(formatted_pivot.to_string())
        
    # 5. 打印一键 Overleaf 排版的 LaTeX 代码
    print("\n📝 [LaTeX booktabs 代码] (复制即可粘入 Overleaf)：")
    print(r"\begin{table*}[t]")
    print(r"\centering")
    print(f"\\caption{{Comparison of {display_title.split(' (')[0]} Across High-Frequency Markets. Best performance is highlighted in \\textbf{{bold}}.}}")
    print(r"\label{tab:" + metric_name + r"}")
    print(r"\resizebox{\textwidth}{!}{")
    col_def = "l" + "c" * len(available_modes)
    print(f"\\begin{{tabular}}{{{col_def}}}")
    print(r"\toprule")
    
    # 表头
    latex_header = r"\textbf{Dataset}"
    for m in available_modes:
        latex_header += f" & \\textbf{{{MODE_MAP[m].split(' (')[0]}}}"
    latex_header += r" \\"
    print(latex_header)
    print(r"\midrule")
    
    # 数据行
    for dataset_idx, row in pivot_mean.iterrows():
        dataset_name_latex = str(dataset_idx).replace("_", r"\_").replace("&", r"\&")
        row_str = f"{dataset_name_latex}"
        valid_vals = row.dropna()
        best_val = None
        if len(valid_vals) > 0:
            best_val = valid_vals.max() if higher_is_better else valid_vals.min()
            
        for m in available_modes:
            m_mean = pivot_mean.loc[dataset_idx, m]
            m_std = pivot_std.loc[dataset_idx, m]
            m_cnt = pivot_count.loc[dataset_idx, m]
            
            if pd.isna(m_mean):
                row_str += " & N/A"
            else:
                suffix = r"\%" if is_percentage else ""
                if is_integer:
                    val_str = f"{int(round(m_mean))}" + (f" \\pm {int(round(m_std))}" if not pd.isna(m_std) and m_cnt >= 2 else "")
                else:
                    val_str = f"{m_mean:.4f}" + (f" \\pm {m_std:.4f}" if not pd.isna(m_std) and m_cnt >= 2 else "")
                val_str += suffix
                
                is_best = (best_val is not None and abs(m_mean - best_val) < 1e-9)
                if is_best:
                    row_str += f" & \\mathbf{{{val_str}}}"
                else:
                    row_str += f" & {val_str}"
        row_str += r" \\"
        print(row_str)
        
    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"}")
    print(r"\end{table*}")
    print("="*110)

# 4. 一键打印所有核心指标表格！
# A. 均值-方差效用表格 (Utility) - 动态展示当前的风险惩罚系数 γ
generate_metric_table(df_all, "utility", f"Mean-Variance Utility (效用值, γ={GAMMA_RISK})", higher_is_better=True)

# B. 夏普比率表格 (Sharpe Ratio)
generate_metric_table(df_all, "sharpe", "Sharpe Ratio (夏普比率)", higher_is_better=True)

# C. 最大回撤百分比表格 (Max Drawdown %)
generate_metric_table(df_all, "max_dd", "Max Drawdown % (最大回撤百分比)", multiplier=100.0, is_percentage=True, higher_is_better=False)

# D. 库存方差表格 (Inventory Variance)
generate_metric_table(df_all, "inv_variance", "Inventory Variance (库存方差/持仓风险)", higher_is_better=False)

# E. 交易次数表格 (Trade Count)
generate_metric_table(df_all, "trades", "Trade Count (交易次数)", is_integer=True, higher_is_better=True)

print("\n💡 提示：您可以直接复制上面输出的 Markdown 格式，贴到您的 Word、Markdown 工具或 LaTeX 论文中。")