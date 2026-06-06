Jiawei Qian
+86 18983425855 | jiaweiqian0@gmail.com | Shanghai, China

AI-Native Agent / LLM Systems Engineer currently serving as the end-to-end owner of the core intelligent experience of an AI Browser — accountable for agent quality iteration, deep-scenario definition, core-case design, and next-generation product intelligence evolution, driving both the current Agent Harness self-iteration pipeline (auto-evaluation → auto-analysis → auto-iteration) and the definition of the next-generation "Browser × Agent" product form and capability boundary. Work methodology: decompose and design agent loops through a control-theoretic lens (policy as plant, harness as controller, evaluator / semantic observer as feedback sensor), and reframe capability iteration as a training process — treating benchmark scores as the objective, structured failure attribution as the loss, and context / memory / skill / system prompt as the updatable parameter space on which a higher-order agent applies gradient-like iterative updates. Experienced in restructuring complex R&D work into AI-executable subproblems and designing human-AI collaborative workflows that reduce manual effort and accelerate experimentation and quality iteration.

## PROFESSIONAL EXPERIENCE

### AI Agent Engineer, Meituan
Dec 2025 - Present

End-to-end owner of the core intelligent experience of an AI Browser, simultaneously accountable for agent quality iteration, deep-scenario definition, core-case design, and next-generation product intelligence evolution. Bottom-up: own agent runtime architecture, the Agent Harness self-iteration pipeline, and the trajectory-data flywheel. Top-down: define the product form, capability boundary, and experience baseline of the next-generation "Browser × Agent", covering browser-use / computer-use scenarios.

#### Product & Experience Ownership (AI Browser Core Intelligence)
- Act as the end-to-end owner of the AI Browser's core intelligent experience, aligning three tracks — agent quality, product form, and capability evolution — by externally defining the next-generation "Browser × Agent" interaction paradigm, deep-scenario boundary, and experience baseline, and internally decomposing product goals into benchmarks, case sets, and attribution signals that the self-iteration system can directly consume.
- Led deep-scenario definition and core-case design: abstracted a measurable, regression-ready set of high-value tasks from real user journeys (long-horizon information gathering, cross-tab / cross-site coordination, form and transactional operations, complex web understanding), used as both the setpoint of agent capability and the anchor of product capability.
- Drove the evolution roadmap of product intelligence: mapped "model capability upgrade / harness architecture evolution / skill distillation / context & memory mechanisms" to concrete product capability milestones (usable → stable → predictable → scalable), so that every agent quality iteration translates into a perceivable product experience upgrade rather than a purely internal metric improvement.

#### Methodology
- Designed the agent loop through a control-theoretic lens: treat the LLM policy as the plant, the harness as the controller, and the evaluator / semantic observer as feedback sensors, with explicit separation of setpoint (task goal), state (execution context), observation (environment feedback), and actuation (tool calls), suppressing oscillation and divergence in long-horizon tasks and shifting robustness from single-point prompt tuning to system-level guarantees.
- Reframed capability iteration as a training process: treat benchmark scores as the objective, structured auto-analysis outputs as the loss, and context structure, memory organization, agent skills, and system prompts as the updatable "parameter space" over which a higher-order agent applies gradient-like iterative updates, making every benchmark regression an interpretable "parameter step" rather than ad-hoc manual tuning.

#### Browser-Use & Computer-Use Agent Runtime
- Led AI-native agent capability iteration by identifying high-value problems, filtering out low-leverage directions, decomposing complex R&D goals into tool-assisted AI-executable tasks, and redesigning workflows to reduce manual involvement and accelerate experimentation.
- Designed and iterated a ReAct-loop-based agent runtime, covering state management, observation abstraction, tool-calling protocols, and failure recovery, improving execution stability for long-horizon tasks.
- Drove the decoupling of "soft optimization" and "hard constraints" in the agent harness by introducing a State Tracker and Semantic Observer, shifting system robustness from model-dependent behavior to architecture-backed guarantees and reducing logical oscillation in complex tasks.
- Designed an orthogonal and atomic toolset spanning the full chain from perception to execution, reducing semantic ambiguity during tool selection and improving decision reliability and system maintainability.
- Built multi-level context denoising and prompt optimization mechanisms for high-noise observation settings, improving action compliance and task completion quality under perceptual saturation.

#### Auto-Iteration, Agent Skill & Trajectory Data Flywheel
- Developed Playbook / Agent Skill capabilities that automatically trace back critical paths, UI elements, and action sequences from execution trajectories, distilling them into reusable skills for test-time reuse and self-evolution.
- Built a full closed-loop pipeline of auto-run -> auto-evaluation -> skill extraction / skill-based re-run -> auto-iteration, enabling the system to automatically evaluate outcomes, attribute failures, distill reusable skills, and iterate on task performance from execution feedback and trajectory data.
- Explored an SFT data flywheel based on real expert trajectories, building lightweight-model alternatives for vertical tasks and validating the replacement potential of 7B / 30B models in selected high-frequency scenarios to reduce token cost and end-to-end latency.

### Software Engineer, Microsoft
May 2024 - Present

#### Agent Workflow, Code Generation & Memory Optimization
- Advanced the transition of analytics and development workflows from manual execution to AI-native pipelines powered by agents, code generation, and validation, reducing human operations through task decomposition, automated checks, and iterative retry loops.
- Built data analysis agents with GPT-4o, LangGraph, and OpenManus / MetaGPT, enabling the core flow from user query to automated data analysis and visualization, with support for RAG, domain knowledge graphs, and internal analytics tools.
- Iterated on a closed loop for SQL-to-DataFrame code generation with sandbox validation and regeneration, improving the executability and reliability of generated analytics code.
- Applied MemGPT-style layered context management for external knowledge, chat history, and tool feedback, reducing token usage per task by 30% and improving performance on multi-turn analytical tasks.

#### LLM Evaluation & Data Flywheel Infrastructure
- Built Azure ML evaluation pipelines with GPT-4o, o1, and DeepSeek V3 / R1, combining user signals to generate long-term profiles and automatically evaluate and monitor ranking and feed relevance for MSN / Bing scenarios at a scale of about 416M tokens per day.
- Designed cache and parallelization strategies for embedding and text-processing pipelines, using a hybrid LRU / LFU cache mechanism to achieve a 95%+ hit rate, reduce quota usage by 90%, cut component runtime by 60%, and improve pipeline robustness.
- Continuously optimized prompts and evaluation metrics on top of a golden set, improving precision and recall in relevance evaluation and supporting a closed loop for data labeling and iteration.

#### Multi-Modal LLM Pipeline & Data Infrastructure
- Built multi-model content understanding and monitoring pipelines for text, image, and video, supporting feed-quality analysis in recommendation scenarios across dynamic prompting, cache reuse, and pipeline optimization.
- Developed large-scale data pipelines with ClickHouse and Spark to aggregate user, content, and request-signal data; improved parallelism by restructuring connection logic and dependencies, reducing the core SLA from 5 days to 4 days while adding input/output monitoring to improve data stability.

### AI Algorithm Engineer, Huawei
July 2022 - May 2024

#### Modeling & Training
- Researched GRU / Transformer-based sequence modeling and entropy coding methods, using temporal probability modeling for NLP data compression and vector quantization for model parameter compression, improving coding efficiency from 6 bits to 5 bits and increasing compression ratio by about 20%.
- Contributed to the modeling and training of end-to-end deep learning communication systems; designed loss functions based on real phase-noise and power-amplifier distortion data and trained neural transceivers whose transmission performance approached the theoretical optimum under AWGN.

#### Optimization & Approximation
- Designed parallel approximate solvers for NP-hard resource allocation problems using randomized algorithms, greedy strategies, and selection mechanisms, then ensembled results with bagging / expert-style methods to achieve about 95% of near-optimal integer programming solutions while reducing solve time from 1500 seconds to 3 seconds.

## EDUCATION

### University of Waterloo
2021 - 2022
MEng, Electrical & Computer Engineering

### Chongqing University
2015 - 2020
BEng, Electrical Engineering and Automation

## SKILLS

### Languages
Python, Golang, SQL

### AI-Native Engineering
Problem scoping and prioritization, AI task decomposition, human-in-the-loop workflow redesign, Cursor, Codex

### LLM / Agent
Control-theoretic agent loop design (plant / controller / observer / feedback), training-style iteration (benchmark -> loss -> gradient over context / memory / skill / system prompt), agent harness self-iteration pipeline, auto-evaluation / auto-analysis / auto-iteration, browser-use / computer-use agents, ReAct, LangGraph, OpenManus / MetaGPT, RAG, prompt engineering, context / memory optimization, tool orchestration, skill injection & design, agent evaluation

### Training / Modeling
SFT, trajectory data construction, Transformers, GRU, CNN, model compression, offline evaluation, approximation optimization

### Data / Infrastructure
Spark, ClickHouse, Azure ML, MySQL, Kafka, CI/CD, RESTful APIs

### Language
English, TOEFL 104, GRE 330

## PUBLICATION

Transformer-based Fuzzer | EAI MONAMI 2022

## AWARDS

- First Prize, Chinese Mathematics Competitions for College Students
- Top 5 Coder, Huawei ICT Department (Python and Java)
- Huawei Future Star
