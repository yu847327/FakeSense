# 目录结构+分工

```
FakeSense/
├── generate/
│   └── generate_prompts.py         # 生成对抗内容
├── detect/
│   ├── detect_perspective.py       # 使用 Perspective API 检测
│   └── detect_tisane.py            # 使用 Tisane.ai 检测
├── evaluate/
│   └── score_risk.py               # 风险打分逻辑与分析
├── utils/
│   └── prompt_templates.json       # 存储各种prompt模板
├── report/
│   └── final_report.ipynb          # 统计与可视化分析
├── run_demo.py                     # 一键运行整套流程
└── README.md
```

| 成员角色                   | 任务内容                                                                     | 目录与文件                                                       |
| ---------------------- | ------------------------------------------------------------------------ | ----------------------------------------------------------- |
| 🧠 **组长 / 统筹（组员A）**    | - 搭建项目骨架（目录结构）<br>- 协调分工与进度<br>- 维护 `README.md`<br>- 集成 `run_demo.py` 脚本 | `README.md`<br>`run_demo.py`<br>总结构管理                       |
| ✍️ **Prompt 工程师（组员B）** | - 编写高质量的 prompt 模板<br>- 组织主题分类（如煽动性、偏见、误导性等）<br>- 管理 JSON 格式             | `utils/prompt_templates.json`<br>配合 generate 模块             |
| ⚙️ **内容生成模块（组员C）**     | - 编写自动调用 LLM 的脚本，生成对抗性文本<br>- 支持多 prompt/多主题批量生成                         | `generate/generate_prompts.py`                              |
| 🧪 **检测模块（组员D）**       | - 封装对 Perspective API、Tisane.ai 的调用<br>- 输出毒性、偏见等检测结果                    | `detect/detect_perspective.py`<br>`detect/detect_tisane.py` |
| 📊 **风险评估模块（组员E）**     | - 对检测结果进行量化打分<br>- 构造“热搜潜力评分”模型<br>- 分析内容风险等级                            | `evaluate/score_risk.py`                                    |
| 📈 **结果展示与报告（组员F）**    | - 汇总各阶段结果<br>- 可视化图表、统计信息<br>- 整理 `final_report.ipynb` 报告                | `report/final_report.ipynb`                                 |

