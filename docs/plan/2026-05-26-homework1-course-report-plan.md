# Homework1 课程报告 PPT 计划

## Summary

使用 `Presentations` 插件按 `template-following` 模式，基于模板 `7bb5add2-3726-4fcb-b4c6-730f25f6141a.pptx` 制作一份 `9` 页的中文课程汇报 PPT，主题为 `Homework 1: SLU Intent Detection`。
报告正式沿用现有实验产物 `homework1/results/best_config.json` 与 `homework1/results/grid_search_results.json`，不等待补跑。
封面只放课程与作业信息，不放个人署名。
最终导出文件名定为 `homework1-course-report.pptx`。

## Implementation Changes

- 严格走 `Presentations` 的模板复制编辑流程，不重建新模板、不改视觉系统、不换字体配色。
- 执行第一步先跑 `inspect_template_deck.mjs` 做模板审计；如果同样触发当前已观察到的 `artifact-tool/skia-canvas` 渲染崩溃，则立即停止并报告插件运行时 blocker，不改用 Python、LibreOffice 或手工 OOXML 旁路。
- 在插件工作区内生成并维护 `template-audit.txt`、`template-frame-map.json`、`deviation-log.txt`，并通过 starter deck 复制源页后原位改字。

### Slide Map

1. `sourceSlide 2` 封面
   内容：课程名、`Homework 1` 标题、主题副标题 `NumPy-based Chinese SLU Intent Detection`、日期。
2. `sourceSlide 3` 目录
   目录固定为：任务背景、方法设计、实验设置、结果分析、总结反思。
3. `sourceSlide 10` 作业概览
   三块内容分别写：任务目标、数据集规模、模型方案。
   必含数据：`train=162`、`dev=3200`、`test=4000`、`34` 个训练标签、手写 softmax regression。
4. `sourceSlide 18` 方法流程
   四步流程固定为：文本清洗与分词、特征提取、softmax 训练、验证与提交生成。
   文案基于 `homework1/main.py` 和 `homework1/src/data_loader.py`。
5. `sourceSlide 11` 实验设置
   四个卡片固定写：特征空间、优化器、正则与 batch、搜索规模。
   必含结论：`4` 类特征、`2` 个优化器、`3` 个学习率、`3` 个 L2、`2` 个 batch size，共 `144` 组组合。
6. `sourceSlide 17` 关键结果
   作为主结果页，放最优配置与特征对比。
   必含最优配置：`char_wb_tfidf`、`ngram 2-5`、`max_features 12000`、`adam`、`lr 0.03`、`l2 1e-4`、`batch 16`、`best_epoch 22`、`val_acc 0.23`。
   同页展示各特征最佳验证准确率：`bow 0.1903`、`ngram 0.1925`、`tfidf 0.2200`、`char_wb_tfidf 0.2300`。
7. `sourceSlide 14` 结果分析
   三栏固定写：为什么字符级 TF-IDF 更优、优化器影响、数据集限制。
   必提事实：Adam 峰值略优；训练集极小且类别覆盖不完整，`train` 独有标签 `咨询（含查询）电商货品信息` 未出现在 `dev`。
8. `sourceSlide 15` 总结与反思
   四块固定写：已完成工作、实验收获、当前不足、可改进方向。
   改进方向只写与作业直接相关的最小延伸：补足训练样本、改进分词/特征、做类别不平衡处理、与 `homework2` 的 BERT 方案形成对照。
9. `sourceSlide 19` 结束页
   仅保留感谢语，清理所有无关占位。

## Test Plan

- 运行模板审计、starter deck 复制、最终 `check_template_fidelity.mjs`，确保每一页都来自映射源页且是原位编辑。
- 检查最终 PPTX 不存在空占位符、默认提示词、缺失页码、缺失页脚、字体溢出或文本裁切。
- 逐项核对数据来源：
  - 数据规模来自 `nlp-text-classification-experiments/*.csv`
  - 配置与最佳结果来自 `homework1/results/*.json`
  - 方法流程来自 `homework1` 源码与 README
- 导出预览图后人工过一遍 `9` 页节奏，确认封面、目录、结果页、总结页主阅读顺序清晰。

## Assumptions

- 报告语言默认中文，专业名词可保留英文，如 `softmax regression`、`TF-IDF`、`Adam`。
- 不补个人姓名、学号、班级，封面只保留课程与作业信息。
- 当前 `homework1/results/` 视为正式实验依据；由于本地 Python 环境缺 `jieba`，本轮不以补跑结果作为前置条件。
- 不新增外部图片、图标或校徽资源，只复用模板已有视觉资产。
