# Voltage Extend GUI

轻量说明与使用要点，包含最近的 dataclean 集成、延迟导入优化与低频扰动功能（已简化版）。

## 快速启动
1. 进入项目目录：
   ```
   cd /Users/hive/Desktop/氢能产品数据/voltage-extend-gui
   ```
2. 建议使用虚拟环境并安装依赖（Mac）：
   ```
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```
3. 启动程序（Mac）：
   ```
   python3 -u src/main.py
   ```

## 主要改动摘要
- dataclean 已集成到 `src/process.py`：
  - process.clean_raw_data(df, col='Voltage')：基于 IQR 检测异常并按“顺移填补”实现填值。
  - load_data 会在读取后调用 clean_raw_data（若可用）。
- 不再做自动插值：
  - process.process_data 返回的 `Voltage_interpolated` 等于清洗后 `Voltage`（保留原始点连线）。
- 启动性能优化：
  - 延迟导入 matplotlib/scipy/sklearn（按需导入），PlotWidget 延迟创建画布。
- GUI 改动：
  - 移除部分调参控件（MAD/tol_frac/abs_cap），增加：进度对话、手动生成预测、为预测线添加低频扰动（可选幅度、默认 0.2%），扰动不修改原始数据且保留端点不变。

## 依赖说明
- 必需（GUI 运行最低依赖）：
  - Python 3.8+
  - pandas
  - numpy
  - PyQt5
  - openpyxl (读取 .xlsx)
- 可选（某些算法或功能按需启用）：
  - matplotlib（绘图，若不安装 GUI 中会降级不显示画布）
  - scikit-learn（KMeans，用于分段判别，缺失时降级为基于中位数的分段）
  - scipy（savgol_filter，用于趋势估计，缺失时使用滑动均值）

请优先安装 requirements.txt（包含推荐版本）；若不需要某些可选功能，可在虚拟环境中移除对应包。

## 调试
- 查看 dataclean 统计（读取后）：
  - processed_df.attrs 中可能包含 dataclean_* 统计信息。
- 若启动慢：已加入按需导入与 importtime 诊断建议（见源代码注释）。

## 开发与提交
- 修改后可使用：
  ```
  git add -A
  git commit -m "chore: update README and requirements"
  git push origin HEAD
  ```