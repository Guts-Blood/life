钱家伟
+86 18983425855	jiaweiqian0@gmail.com	Shanghai, China
AI Agent Systems Engineer，当前作为 AI Browser 核心智能体验端到端 owner，同时负责 Agent 效果迭代、深度场景定义、核心 case 设计与下一代产品智能能力演进——既驱动当前 Agent Harness 自迭代系统（auto-evaluation → auto-analysis → auto-iteration 全链路）落地，也定义"Browser × Agent"下一代产品形态与能力边界。工作方法论以控制论视角拆解 Agent Loop（plant / controller / observer / feedback），并以训练范式重构效果迭代——把 benchmark score 视作 objective，把归因信号视作 loss，把 context / memory / skill / system prompt 视作可更新参数空间，在其上施加"gradient-like"的迭代更新。兼具 Agent runtime 架构、工具链设计、产品能力定义与 training flywheel 落地经验。
PROFESSIONAL EXPERIENCE
AI Agent Engineer, Meituan
Dec 2025 – Present
作为 AI Browser 核心智能体验的端到端 owner，同时负责 Agent 效果迭代、深度场景定义、核心 case 设计与下一代产品智能能力演进——向下负责 Agent Harness 自迭代系统（auto-evaluation → auto-analysis → auto-iteration 全链路）、Agent runtime 架构与 training flywheel，向上定义"Browser × Agent"的产品形态、能力边界与体验基线，覆盖 browser-use / computer-use 场景。
Product & Experience Ownership（AI Browser 核心智能体验）
•	作为 AI Browser 核心智能体验端到端 owner，统一拉通 Agent 效果、产品形态与能力演进三条线：对外定义下一代"Browser × Agent"的用户交互范式、深度场景边界与体验基线，对内将产品目标拆解为可被迭代系统消费的 benchmark、case 集合与归因信号。
•	主导深度场景定义与核心 case 设计：从真实用户旅程中抽象出可衡量、可回归的高价值任务集（长路径信息获取、跨 tab / 跨站点协同、表单与事务类操作、复杂网页理解等），作为 Agent 能力的 setpoint 与产品能力的锚点。
•	推动产品智能能力演进路线：将"模型能力升级 / harness 架构演进 / skill 沉淀 / context & memory 机制"映射到具体产品能力节点（可用 → 稳定 → 可预期 → 可放大），让每一次 Agent 效果迭代都对应到一次可感知的产品体验升级，而非纯内部指标优化。
Methodology（方法论）
• 以控制论视角设计 Agent Loop：把 LLM policy 当作 plant，把 harness 当作 controller，把 evaluation / semantic observer 当作 feedback sensor，通过显式拆分 setpoint（任务目标）、state（执行上下文）、observation（环境反馈）与 actuation（工具调用），抑制长路径任务中的振荡与发散，换取系统级稳定性而非单点 prompt 调优。
• 以训练范式重构效果迭代：将 benchmark score 视为 objective，auto-analysis 产出的结构化归因当作 loss，把 context 结构、memory 组织、Agent Skill 与 system prompt 视作可更新的"参数空间"，由高阶 Agent 控制层在其上施加 gradient-like 的迭代更新，使每次 benchmark 回归都等价于一次可解释的"参数步进"，而非一次性人工调参。
Agent Harness Self-Iteration Pipeline（核心方向）
• 设计并落地 auto-evaluation → auto-analysis → auto-iteration 全链路闭环系统：以更高阶的 Agent 控制层自动完成子 Agent 的效果评测、失败归因与迭代决策，将"人驱动迭代"升级为"系统驱动迭代"，大幅降低人工参与带宽。
• 拆解 Agent 效果迭代的完整可调空间并实现自动化迭代覆盖：context 结构与去噪策略、tool 空间定义与组合、system prompt 工程、Agent Skill 注入与编排、以及方案选型（GUI Agent vs CLI Agent vs 混合模式），使每个维度均可被迭代系统独立感知、评估与优化。
• 构建 Skill 自动沉淀与注入机制：从执行轨迹中自动回溯关键路径、UI 元素与动作序列，沉淀为可复用 Playbook / Agent Skill，支持 test-time skill reuse 与 self-evolution，形成 skill 维度的自迭代飞轮。
• 设计迭代系统的 evaluation 层：区分 rule-based 判定与 LLM-as-judge 评估，结合任务目标达成、轨迹效率与行为合规性多维度指标，为 auto-analysis 提供结构化归因信号而非简单通过率。
• 设计迭代系统的 analysis 与 decision 层：基于评测信号自动定位瓶颈维度（context 不足 / tool 缺失 / prompt 歧义 / skill 缺口），生成可执行的迭代提案并自动应用，实现迭代闭环。
Browser-use & Computer-use Agent Runtime
• 设计并迭代基于 ReAct loop 的 Agent runtime，负责状态管理、观测抽象、工具调用协议与失败恢复，提升长路径任务中的执行稳定性。
• 推动 Agent harness 中"软调优"和"硬约束"解耦，引入 State Tracker 与 Semantic Observer，将系统鲁棒性从依赖模型能力转向依赖架构保障，抑制复杂任务中的逻辑振荡。
• 设计正交化、原子化工具集，覆盖感知到执行全链路，降低 tool selection 阶段的语义歧义，提升决策确定性与系统可维护性。
Training Flywheel
• 探索基于真实专家轨迹的 SFT 数据飞轮，构建轻量模型在垂直任务中的替代方案，验证 7B / 30B 模型在部分高频场景下对高成本模型调用的替代潜力，以优化 token cost 与端到端时延。

Software Engineer, Microsoft
May 2024 – Present
负责 Agent 驱动的数据分析系统、LLM evaluation pipeline 与大规模数据基础设施建设，持续覆盖 agent workflow、代码生成、memory/context optimization 以及 cost/performance optimization。
Agent Workflow, Code Generation & Memory Optimization
•	推动分析与研发流程从人工执行转向 Agent + code generation + validation 驱动的 AI-native workflow，通过任务拆解、自动验证与迭代重试减少人工操作并提升交付效率。
•	基于 GPT-4o、LangGraph 与 OpenManus / MetaGPT 开发数据分析 Agent，打通 User Query -> Agent -> 数据分析与可视化的核心流程，支持结合 RAG、领域知识图谱和内部分析工具完成自动化分析与报告生成。
•	迭代开发 SQL -> DataFrame 代码生成与 sandbox validation -> re-generate 闭环，提升分析代码生成的可执行性与可靠性。
•	基于 MemGPT 思路对 external knowledge、chat history 和 tool feedback 进行 context 分层管理，减少 30% token usage per task，并提升多轮分析任务表现。
LLM Evaluation & Data Flywheel Infrastructure
•	基于 GPT-4o、o1、DeepSeek V3 / R1 等模型构建 Azure ML 评估管道，结合用户信号生成长期画像，并对 MSN / Bing 等场景中的 ranking 与 feed relevance 进行自动化评估和监控，处理规模约 416M tokens/day。
•	围绕 embedding 与文本处理链路设计缓存与并行优化策略，采用 LRU / LFU 混合缓存机制，实现 95%+ hit rate，减少 90% quota usage 与 60% 组件运行时间，并提升 pipeline 鲁棒性。
•	基于 golden set 持续进行 prompt 调优与评测指标优化，提升 relevance evaluation 的 precision / recall 表现，支撑数据标注与迭代闭环。
Multi-modal LLM Pipeline & Data Infrastructure
•	构建多模型内容理解与监控 pipeline，对文本、图像和视频内容进行理解、分类与质量监控，辅助推荐场景中的 feed 质量分析，覆盖 dynamic prompt、cache reuse 与 pipeline 优化等核心环节。
•	基于 ClickHouse 与 Spark 搭建大数据管道，聚合用户、内容与请求信号数据；通过重构连接逻辑与依赖关系提升并行度，将核心 SLA 从 5 天缩短到 4 天，并补充输入输出监控机制以提高数据稳定性。
AI Algorithm Engineer, Huawei
July 2022 – May 2024
负责训练、建模与复杂优化问题求解，侧重 sequence modeling、模型压缩与近似优化，为后续 Agent / LLM 系统工作提供训练与算法基础。
Modeling & Training
•	研究基于 GRU / Transformer 的序列建模与熵编码方案，利用时域概率建模进行 NLP 数据压缩，并结合向量量化压缩模型参数，将编码效率从 6 bits 提升到 5 bits，压缩率提升约 20%。
•	参与端到端深度学习通信系统建模与训练，基于真实相位噪声与功率放大器失真数据设计损失函数并训练神经网络收发机，使传输性能接近 AWGN 理论最优。
Optimization & Approximation
•	针对 NP-hard 资源分配问题，设计随机算法、贪婪策略与选择机制的并行近似求解方案，并结合 bagging / expert 思想进行结果集成，在接近最优整数规划解 95% 的同时，将求解时间从 1500 秒降至 3 秒。
EDUCATION
University of Waterloo
2021-2022
MEng, Electrical & Computer Engineering
Chongqing University
2015-2020
BEng, Electrical Engineering and automation
SKILLS
Languages:
Python, Golang, SQL
AI-native Engineering:
Problem scoping & prioritization, AI task decomposition, human-in-the-loop workflow redesign, Cursor, Codex
LLM / Agent:
Agent harness self-iteration pipeline, auto-evaluation / auto-analysis / auto-iteration, control-theoretic agent loop design (plant / controller / observer / feedback), training-style iteration (benchmark→loss→gradient over context / memory / skill / system prompt), browser-use / computer-use agents, ReAct, context engineering, tool orchestration, system prompt engineering, skill injection & design, LangGraph, RAG, prompt engineering
Training / Modeling:
SFT, trajectory data construction, Transformers, GRU, CNN, model compression, offline evaluation, approximation optimization
Data / Infrastructure:
Spark, ClickHouse, Azure ML, MySQL, Kafka, CI/CD, RESTful APIs
Language:
English, TOEFL 104, GRE 330

PUBLICATION
Transformer-based Fuzzer	EAI MONAMI 2022
AWARDS
•	中国大学生数学竞赛：一等奖
•	华为公司ICT部门Top 5 Coder（Python和Java）
•	华为"未来之星"
