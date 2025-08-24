# Voltage Extend GUI

轻量说明与使用要点，包含最近的参数暴露与首周期截断行为（已简化版）。

## 快速启动
1. 进入项目目录：
   ```
   cd /Users/hive/Desktop/氢能产品数据/voltage-extend-gui
   ```
2. 启动程序（Mac，已安装依赖）：
   ```
   python3 -u src/main.py
   ```

## 本次重要改动（简短）
- 将 dataclean 功能集成到 process 模块：
  - process.clean_raw_data(df, col='Voltage')：基于 IQR 检测异常并按“顺移填补”实现填值。
  - load_data 会在读取后调用 clean_raw_data（若可用）。
- 不再对数据做自动插值：
  - process.process_data 现在保留原始清洗后的 Voltage 作为 Voltage_interpolated（不进行额外插值或平滑）。
- 首周期衔接：
  - 新增 strict_truncate 标志（GUI 默认开启），可控制首周期是否严格截断（避免相位重排）。
- 启动/性能优化：
  - 延迟导入 matplotlib/sklearn/scipy（按需导入），PlotWidget 延迟创建画布，减小启动开销。
- GUI 简化：
  - 已移除 MAD/tol_frac/abs_cap 的可视调参控件（参数仍存在于 process 的默认值中：cap_k=10, abs_cap_mult=20）。

## 调试与常见操作
- 若需查看 dataclean 统计：
  - 在读取后可以查看 processed_df.attrs 中的 dataclean_in_count / dataclean_outliers / dataclean_out_count。
- 若需增加/调整限幅参数，请修改 process.py 中 synthesize_extension_from_samples 的 cap_k / abs_cap_mult 参数值或在后续版中把控件恢复到 GUI。

## 提交/推送
下面命令会把本地改动提交并推送到当前分支（请先确认已 commit 本地更改或按需修改 commit 信息）。
参见项目根目录的 git 命令说明（见下方示例）。