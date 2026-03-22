# 项目检查与可复现性确认

## 1. 结论
- 代码结构完整：`是`
- 主要实验产物齐全：`是`
- 数据处理链路可加载：`是`
- Baseline/消融结果表结构正确：`是`
- GPU 环境可用：`是`

## 2. 当前确认无误的内容
- 原始数据文件存在，且表头与电站 ID 已统一为英文。
- 主实验入口、配置、数据处理、模型与报告模块均存在。
- `baseline_metrics.csv / ablation_metrics.csv / plant_level_metrics.csv` 已生成。
- `baseline_daytime_metrics.csv / baseline_physical_metrics.csv / seed_repeat_summary.csv / rolling_origin_summary.csv` 已生成。
- `README.md`、训练配置记录、SDM 口径说明、参考稿、鲁棒性记录和主要图表已经生成。
- 代码可完成数据加载、特征工程与监督样本构建。

## 3. 可复现性说明
- 当前环境：Python 3.12 + CUDA 可用 + GPU 版 PyTorch。
- 重新安装建议命令已在 README 中给出。
- 主实验命令保持为：`./.venv/Scripts/python run_experiments.py`（PowerShell 下实际使用反斜杠路径）。
- 本次检查验证了项目结构、依赖、数据读取和结果文件的一致性。

## 4. 尚需注意的点
- 逐样本预测导出文件已存在，可继续做更细粒度论文图。
- 新版 Hybrid 已改为电站级场景融合，并额外保留固定权重版本作为消融对照。
- StackedXGB 仍是重要对照，但主线 Hybrid 现在具备更强的可解释性。
- 项目现在额外提供 daytime-only、physical violation、多随机种子和 rolling-origin 四类评估结果。
- 训练配置与实际执行轮次已导出为独立表格，可直接写入论文实验设置。

## 5. 数据与特征规模
- 原始样本：`1242372` 行，`40` 列
- 监督样本：`1241216` 行，`111` 列
- 训练集：`992740` 行
- 验证集：`123948` 行
- 测试集：`123952` 行
- 连续特征数：`96`
- One-hot 后特征数：`111`