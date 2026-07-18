# RGA-KD — 完整项目交接文档（诊断 → 改进 → 实验 → 论文）

> **这份文档是给一个全新 agent 的一站式手册。** 你（执行 agent）的任务是：在服务器上独立完成 RGA-KD 的全部前期诊断、根据诊断结果改进方法、跑完所有正式实验、并最终写出论文。本文件是唯一权威入口——按它走，不要重跑 `/idea-discovery`，不要推翻已冻结的 pilot 结论。
>
> **一句话方法**：对每个样本，比较教师与学生在梯度几何里"把它和其他样本关联起来的方式"是否一致——一致=学生已学会教师在该样本上的关系结构（冗余、可跳过），不一致=教师有学生还没掌握的结构（重点蒸馏）；用"样本×样本"的关系矩阵来比，从而绕开"师生参数空间维度不同"的障碍。
>
> **最重要的纪律**：这是一个 **finding-or-kill** 项目。诊断阶段（Phase A）是硬门槛。**如果诊断证伪了核心假设，就如实写一份负面诊断报告并停止，不要硬推到正式实验。** 诚实的负面结果也是合格交付。

---

## 0. 一键续作 prompt（新会话第一条消息粘这个）

```
/goal 按 RGA_KD_PROJECT.md 把 RGA-KD 项目从诊断推进到论文。先读 RGA_KD_PROJECT.md 全文、
CURRICULUM_KD_REVIEW.md（领域定位）、CURRICULUM_KD_DEEP_ANALYSIS.md（方法机制）、
refine-logs/EXPERIMENT_PLAN.md（已有 E0–E6 与 GPU 协调规则）、pilot/pilot_gradgeom.py（可复用
基础设施）。不要重跑 /idea-discovery，不要推翻冻结的 pilot 结论。

按 Phase A → B → C → D 顺序执行，每个 Phase 的决策门槛（§5–§8）是硬性的：
- Phase A 诊断若证伪核心假设（§5 的 STOP 条件），写负面诊断报告并停止，不进 Phase C。
- 每次 CUDA 启动前必须按 §9 做 nvidia-smi 检查 + CUDA_VISIBLE_DEVICES pin + 记录 gpu_uuid。
- 每跑完一步，更新 §11 的进度日志和 refine-logs/EXPERIMENT_TRACKER.md。
- 所有结果必报种子方差；端到端 wall-clock 必须含选择成本。

当前状态见 §11 进度日志。从第一个未完成的步骤继续。
```

---

## 1. 背景与定位（30 秒读完）

**这是什么**：RGA-KD 是 GradSpan-KD 项目（本仓库主线）在"师生对齐"方向上的方法升级。GradSpan-KD 的核心主张是 *KD 数据选择应在学生参数梯度几何空间做*；RGA-KD 把"难度信号"从"KD 梯度幅值/子空间"升级为"**师生梯度关系结构是否对齐**"。

**为什么做这个方向**（文献依据，详见 `CURRICULUM_KD_REVIEW.md`）：
- 纯"按难度排序"是被大规模证伪的弱杠杆（When-Do-Curricula-Work, Rethinking-Easy-to-Hard）。
- 几乎所有现有方法的难度信号都停在**输出/损失标量空间**；理论（Panigrahi, ICLR'25）和实证（PACED, 2026）都指向**参数梯度几何**才是 KD 有用信号的所在地，而那里基本无人占据。
- 导师明确要往"**师生 align**"方向想。RGA-KD 用关系/相对对齐（Gram/CKA 不变性）严格化解了"两个参数空间维度不同"的障碍，**既守住导师方向、又有真 novelty**。

**冻结的 pilot 既有证据**（`pilot/PILOT_RESULTS.json`，不要重做）：
- KD per-sample 梯度强低秩：1200 维里有效秩 ≈ 23。
- D-optimal 子空间覆盖 coreset 在激进预算（≈1–2× 梯度秩）比随机高 +8.8 acc、种子方差低 ~8×。
- 两条已证伪、必须丢弃的旧主张：「KD 比 CE 更低秩」（打平）；「结构 ≠ 难」（相关 +0.59）。

---

## 2. 方法全规格：RGA-KD（Relational Gradient Alignment for KD）

### 2.1 记号
- 教师 `T`（冻结）、学生 `S`；学生参数 `θ_S`。
- per-sample 学生 KD 梯度：`g_S(x) = ∇_{θ_S} KL(p_T(x) ‖ p_S(x))`。注意它 = `J_S(x)^T (p_S − p_T)`，`J_S` 是学生 logit 对参数的雅可比。
- per-sample 教师签名：`g_T(x) = J_T(x)`（教师 logit 对教师参数的雅可比 = 经验 NTK 特征）。**不能**用教师任务损失梯度（冻结教师≈0、无信号）。
- 随机投影 `Π`（Count-Sketch 或 Rademacher JL），投到 `d` 维（默认 ~4096–8192）。复用 pilot 思路。
- 锚点集 `{a_1,…,a_m}`（m 个样本，默认 m=256–512），用于把 N×N 关系矩阵降成 N×m。

### 2.2 流程
**Step 0｜热身**：学生先在当前蒸馏任务上训几个 epoch（让输出头校准；否则 per-sample 梯度被"任务还没接上"的共同瞬态主导，关系结构被淹没）。pretrained 文本学生需要的步数比从零训的视觉学生少，但都需要。

**Step 1｜取两套投影梯度签名**：对每个候选样本 `x` 与每个锚点，算 `Π g_S(·)`、`Π g_T(·)`。

**Step 2｜建中心化关系矩阵**：
- `K_S[i,j] = ⟨Π g_S(x_i), Π g_S(a_j)⟩`，`K_T` 同理。
- 中心化：`K' = H K H`，`H = I − (1/m)11^T`。

**Step 3｜per-sample 关系残差**（把 pairwise 量变成可选样分数的关键）：
```
r(i) = 1 − corr( K'_T[i,:], K'_S[i,:] )
```
`r(i)` 大 = 师生把 i 放进关系网的方式不同 = 教师有学生没学会的结构 = 重要；小 = 学生已复现 = 冗余。

**Step 4｜教师正确性闸门（必不可少，不是可选项）**：`r(i)` 大也可能是教师在 i 上错了（RAD 陷阱）。剔除 `argmax p_T(x) ≠ y` 的样本，或把分数乘 `p_T(y)`。

**Step 5｜覆盖选样（非 top-k）**：在过闸门的高 `r` 样本里，用 `d_optimal()`（pilot 已实现）在"关系残差向量"上选张得开的子集，避免挑同质 mismatch。

**Step 6｜（可选）排序 + 时效**：选完可用 GraB 式陈旧梯度前缀均衡排序；难度默认静态、隔若干 epoch 重估一次。

### 2.3 数学状态（你必须知道哪些是定理、哪些是待验证假设）
- **站得住的**：Gram 矩阵把维度障碍消掉（恒等事实）；`K=GG^T` 对正交变换不变（"不需对齐参数"的严格依据）；行残差良定义且基无关；eNTK 教师签名、D-optimal 都是标准工具。
- **欠证明、Phase A 要直接检验的假设**：①「关系不匹配 ⟹ 训练价值高」是有动机的猜想、不是定理（无 PACED 那样的最优性证明）；②学生用 KD 梯度、教师用 eNTK 是**类型不一致**——"两边都用 eNTK"才 apples-to-apples 但丢了 KD 动态，必须靠消融定夺；③`r(i)` 依赖锚点选择（Nyström 近似）。

---

## 3. 与已有工作的区别（写 related work 时用，也是 novelty 底线）
- 关系蒸馏 RKD（CVPR'19）、跨网络相似 CKA/SVCCA：都在**表征空间**、用来"蒸"或"度量"，**没人在梯度核空间用关系对齐来选/排 KD 训练样本**。
- Selective-KD（ACL'21）logit 空间逐 token 重加权；TGeo-KD（ICLR'24）输出三角几何重加权；都不是这个。
- LESS（ICML'24）梯度空间但有监督 cos-top-k 无覆盖；TAGCOS（2024）梯度匹配均值；GradSpan-KD（本仓库）KD 梯度幅值/子空间——**RGA-KD 的差异点是"师生关系对齐 + 教师闸门 + D-optimal 覆盖"这个组合，无人做过**。

---

## 4. 总体执行图

```
Phase A 诊断(CPU/小GPU, ~1天)  ──┬─ 通过 ─▶ Phase B 方法定稿  ──▶ Phase C 正式实验  ──▶ Phase D 论文
  (§5 决策门槛)                  │
                                └─ 证伪 ─▶ 写负面诊断报告 + STOP（仍是合格交付）
```

**铁律**：不要在未通过 Phase A 的情况下烧 Phase C 的 GPU 预算。

---

## 5. PHASE A — 诊断（make-or-break，先做这个，便宜）

目标：用最小成本回答"RGA-KD 的核心假设到底有没有信号"。**先在 `pilot/` 规模（sklearn-digits MLP，CPU、几分钟）跑，再在一个真实小 pair（ResNet-56→20 / CIFAR-100 子集）确认。**

### 5.1 要写的脚本：`pilot/pilot_rga_diag.py`
基于 `pilot/pilot_gradgeom.py` 改造（复用它的 teacher/student 构造、per-sample KD 梯度提取、`d_optimal()`、`kd_train()`、effective-rank）。新增：
1. **教师 eNTK 签名** `g_T(x) = J_T(x)`：对每个样本，求教师每个 logit 对教师参数的梯度，拼成雅可比向量（K 类可只取 top-class 或全 logit 拼接；先全 logit）。
2. **随机投影** `Π`：对 `g_S`、`g_T` 各投到 d=2048（pilot 规模可小些）。
3. **锚点 + 关系矩阵**：随机取 m=256 锚点，算中心化 `K'_S`、`K'_T`。
4. **per-sample 残差** `r(i)`，以及全局 `CKA(K_S,K_T)`。
5. **诊断量**（全部写进 `pilot/RGA_DIAG_RESULTS.json`）：
   - `corr(r, kd_loss)`、`corr(r, grad_norm)`、`corr(r, teacher_correct)`；
   - `r` 的分布/结构（是否非平凡：不是全 0、不是纯噪声）；
   - 签名消融：{学生 KD-grad + 教师 eNTK} vs {两边都 eNTK} vs {两边 KD-grad-feature}，各自的 `r` 与上面相关性。
6. **证伪 + 正面对照重训**（用 `kd_train()`，5 seeds，激进预算如 m=1–2×rank）：
   - `select_high_r`（RGA-KD 选中，过闸门 + D-optimal 覆盖）
   - `select_low_r`（专选对齐样本——**证伪臂**）
   - `random`、`hard_highloss`、`d_optimal_grad`（=GradSpan-KD 本体，KD 梯度子空间）
   - 报 mean ± std。

`RGA_DIAG_RESULTS.json` 字段示例（合成值）：
```json
{
  "setup": {"dataset": "sklearn-digits", "N": 1200, "proj_dim": 2048, "anchors": 256},
  "signatures": {
    "studentKD_teacherENTK": {"cka": 0.41, "corr_r_kdloss": 0.18, "corr_r_teachercorrect": -0.33},
    "both_entk":             {"cka": 0.55, "corr_r_kdloss": 0.09, "corr_r_teachercorrect": -0.21}
  },
  "retrain_5seed": {
    "select_high_r":   {"mean": 0.000, "std": 0.000, "n": 40},
    "select_low_r":    {"mean": 0.000, "std": 0.000, "n": 40},
    "random":          {"mean": 0.000, "std": 0.000, "n": 40},
    "d_optimal_grad":  {"mean": 0.000, "std": 0.000, "n": 40}
  }
}
```

### 5.2 决策门槛（硬性，按这个判 GO/STOP/PIVOT）
读 `RGA_DIAG_RESULTS.json`，依次判：

1. **`r` 有结构吗？** 若 `r` 几乎常数（全对齐）或纯噪声（与任何东西都不相关）→ **STOP**：关系残差里没有可选信号。
2. **`r` 只是 KD 损失换皮吗？** 若 `|corr(r, kd_loss)| > ~0.8` → **改签名**（换两边都 eNTK 再看）；若所有签名下都高度相关 → **STOP/PIVOT**：没有超出"按 KD 损失选样"的新信号，退回 GradSpan-KD 幅值版。
3. **证伪臂对吗？** `select_low_r` 必须**明显差于** random（选对齐样本=该跳过的样本）。若 `select_low_r ≥ random` → **核心假设被证伪**，写负面报告 STOP。
4. **正面臂有优势吗？** `select_high_r`（过闸门+覆盖）应在激进预算下 **≥ random 且有竞争力**；理想是 **≥ d_optimal_grad（GradSpan-KD 本体）**。若打不过 GradSpan-KD 本体 → 记录："关系对齐未带来超越幅值/子空间的增量"，考虑 PIVOT 回 GradSpan-KD（仍可把 RGA 作为一个 ablation 臂）。
5. **闸门必要吗？** 若 `corr(r, teacher_correct)` 明显为负（高 r 样本里教师更常出错）→ 证实闸门必要，保留；记录加闸/不加闸的对照。

> **通过条件（进 Phase B/C）**：`r` 有结构 + 不是 KD 损失换皮 + 证伪臂明显更差 + 正面臂至少不输 random（最好 ≥ GradSpan-KD 本体）。
> **任何一条硬性 STOP 触发**：转 §10 写负面诊断报告，结束。

### 5.3 在真实小 pair 上复核
pilot 通过后，在 ResNet-56→ResNet-20 / CIFAR-100 的一个子集（如每类 50 张、热身 5 epoch）上重算 §5.1 的诊断量，确认结论在真实网络上不翻车（lazy/kernel 假设对真实网风险点，必须复核）。GPU 用量小，按 §9 协调。

---

## 6. PHASE B — 根据诊断改进并定稿方法

根据 Phase A 结果做有据的方法选择，把定稿写进 `RGA_KD_METHOD_FINAL.md`：
- **签名选择**：用诊断里"`r` 信号最强、最不像 KD 损失换皮"的那组（studentKD+teacherENTK / both-eNTK / both-KDfeature）。
- **闸门形式**：硬剔除 vs 乘 `p_T(y)`，用诊断对照定。
- **覆盖 vs top-k**：若 D-optimal 覆盖在重训里胜过纯 top-r，保留覆盖；否则简化。
- **静态 vs 动态难度**：默认静态（CLPD/DMC 经验：动态不划算）；若诊断显示 `r` 随训练剧烈漂移，再考虑隔 N epoch 重估。
- **锚点数 m、投影维 d、子空间维 k**：定一组默认值并记录敏感性。
- 若多处证伪 → 方法**降级为 GradSpan-KD + RGA 作为一个选择准则 ablation**，论文叙事相应调整（仍可发，定位为"关系对齐 vs 幅值/子空间"的发现）。

---

## 7. PHASE C — 正式实验

**直接复用 `refine-logs/EXPERIMENT_PLAN.md` 的 E0–E6 骨架与 GPU 协调规则**——RGA-KD 本质是往 E1/E2 里加一个新的选择臂。只在以下处扩展：

### 7.1 固定设置（同 EXPERIMENT_PLAN）
- 视觉对：ResNet-56→ResNet-20 / CIFAR-100（另 WRN-40-2→WRN-16-2）。
- 文本对：BERT-base→BERT-small / SST-2（或 MNLI）。文本是"token/师生几何错配"最真实处。
- 预算轴：先测 KD 梯度有效秩 `R`（E0），按 `m ∈ {0.5R,1R,2R,4R,8R,0.5N,N}` 扫。
- **每格 5 seeds，种子方差是上报指标，不是误差棒附属。**
- 指标：学生 test acc/score、其 std、**端到端 wall-clock（含选择成本与教师软标签推理）**。

### 7.2 E0 — 诊断骨架（~0.5 GPU-day）
算两 pair 的 per-sample KD 梯度矩阵 + 教师 eNTK 签名；Count-Sketch 投影；SVD。报有效秩 `R`（校准预算）、师生关系矩阵 CKA、`r` 分布。

### 7.3 E1 — 空间 / 准则对决 ★GO/NO-GO 硬门★（~2 GPU-days）
固定 D-optimal 与投影维。对比在**激进预算 m∈{1R,2R,4R}**下重训学生：
- **RGA-KD**（关系残差 + 闸门 + 覆盖）
- GradSpan-KD 本体（KD 梯度子空间）
- 晚层特征空间选择、输入显著性空间选择
- random / KD-loss top-k / 不加闸的纯对齐
- 两个模态都跑。**报种子方差。**
- **决策门**：RGA-KD（或 GradSpan-KD，取决于 Phase A）须在 ≥1 模态上以多 seed 清晰 margin 胜过特征空间选择，且 RGA-KD 须 ≥ GradSpan-KD 本体或提供互补价值。**若全部打平 → STOP，转负面诊断 note。** 不通过不得进 E2。

### 7.4 E2 — 主结果锚点（~5 GPU-days）
全预算扫 + native 基线（无稻草人）：random, EL2N, GraNd, high/low-loss, CRAIG, GRAFT(feature MaxVol), TAGCOS(梯度聚类), LESS-style, DPP。报 acc 与种子方差，两模态两 pair。
- **决策门**：RGA-KD 须在激进预算胜过 GRAFT 与 TAGCOS。否则无算法 novelty，降级为 E1 的诊断 note。

### 7.5 E3 — KD 相关性（~2 GPU-days）
同选择法用于 (a) KD/KL 训练与 (b) 纯 CE 训练。比较 RGA-KD 相对 loss-based 选择的**相对**优势在 KD vs CE 是否更大。
- **决策门**：若 KD 与 CE 无区别 → 丢"KD-special"主张，改写成一般激进预算 coreset 论文（较弱）。

### 7.6 E4 — 效率 / 摊销（~1.5 GPU-days）
便宜梯度消融（full / last-layer / LoRA-only / one-step / logit-difference 代理）。**端到端 wall-clock 含选择成本 + 教师软标签推理**。报盈亏平衡点（多少 coreset epoch / 多少被蒸学生时选择成本被摊销）。**关系矩阵 + eNTK 的额外成本必须诚实计入。**

### 7.7 E5 — 方法消融（~2 GPU-days）
投影：Count-Sketch vs Rademacher vs 无。准则：D-optimal vs 岭杠杆分数 vs per-PC 配额 vs 纯 top-r。签名：§6 的三种配对。锚点数 m、子空间维 k、温度、教师选择的鲁棒性。**闸门开/关对照（直接量化 RAD 陷阱）。**

### 7.8 E6 —（可选）教师噪声证伪（~1 GPU-day）
人为污染已知比例的教师 logit，验证它们是否被 RGA-KD 的高 r + 低教师置信识别/剔除——闸门有效性的决定性廉价测试。

### 7.9 Run Order（同 EXPERIMENT_PLAN）
```
E0 ─▶ E1(GATE) ─▶ {E2 ∥ E3 ∥ E4} ─▶ E5 ─▶ E6(opt)
            └─ fail ─▶ 负面/诊断 note 或 kill
```

---

## 8. PHASE D — 写论文

实验决定性后，调用论文写作技能链（仓库已装：`paper-plan → paper-figure → paper-write → paper-compile → auto-paper-improvement-loop`，或一键 `paper-writing`）。

- **定位**：finding paper —— "KD 数据选择的正确信号是师生梯度关系对齐"，RGA-KD 是其实例。**不要包装成纯新算法**（算法 novelty 自评 5–6/10）。
- **claims 用 result-to-claim 技能判定**：结果支持哪条就写哪条，不过度声称。
- **必须呈现**：种子方差；含选择成本的端到端 wall-clock；E1/E5 的反转/闸门消融；与 GradSpan-KD 本体的对照（关系对齐是否带来增量）。
- **诚实写负面**：若某模态打平、若闸门未生效、若 RGA≈GradSpan，如实写进 limitation。
- 写前可选 `/research-review` 或 `/novelty-check` 做独立交叉检查（若服务器装了 Codex MCP）。

---

## 9. 非协商项（贯穿所有 Phase）

1. **共享 GPU 协调**（服务器多租户，A6000 ~48GB）。每个 CUDA `python` 前：
   - `nvidia-smi --query-gpu=index,name,memory.free,memory.used,utilization.gpu --format=csv` 快照；
   - 选 `memory.free` ≥ 阈值的最低 index（E0/单学生 E1 ≥20GB；E2/E3/E4 ≥40GB），`export CUDA_VISIBLE_DEVICES=<idx>`，记录 `gpu_uuid`；
   - 无满足者每 30s 轮询；轮询 >60min 则**暂停**，写进 `EXPERIMENT_TRACKER.md`，转 CPU/写作；
   - 一进程 pin 一卡；每 epoch checkpoint（被别的租户 OOM-kill 时可在别的卡续，损失 ≤1 epoch）；
   - 每次 >10min 的等待、每次 OOM/kill 都记进 `EXPERIMENT_TRACKER.md` 的 GPU 协调日志（含 UUID）。
2. **报种子方差，不只均值。**
3. **端到端 wall-clock 必含选择成本**（含 eNTK/关系矩阵的额外开销）。省算力主张由这个数字决定，不是训练-only 时间。
4. **教师正确性闸门不能省**（否则被教师噪声带崩）。
5. **保留已证伪的教训**：丢「KD 比 CE 更低秩」「结构 ≠ 难」。
6. **诚实负面是合格交付**：诊断或 E1 证伪就如实写 note 并停，不硬推。
7. **硬性资源闸**：任何单次 run 超估时 3× 即 kill；总 GPU-day 超 25 即中止套件并汇报。
8. 每完成一步，更新 §11 进度日志 + `refine-logs/EXPERIMENT_TRACKER.md`。

---

## 10. 负面结果路径（触发 STOP 时怎么交付）

若 Phase A 或 E1 触发 STOP，**不是失败、是合格交付**。写 `RGA_KD_DIAGNOSTIC_NOTE.md`：
- 测了什么、用什么协议、原始诊断数字（含 `RGA_DIAG_RESULTS.json`）；
- 哪条假设被证伪（`r` 无结构 / =KD 损失换皮 / 证伪臂不差 / 打不过 GradSpan-KD）；
- 据此对 GradSpan-KD 主线的建议（多半是"退回 KD 梯度子空间幅值版，RGA 作为 ablation"）；
- 仍可成文：定位为"关系对齐作为 KD 选样信号的负面/边界发现"。

---

## 11. 进度日志（每完成一步就更新这里）

| 日期 | Phase/步骤 | 状态 | 关键结果 / 决策 | 下一步 |
|------|-----------|------|----------------|--------|
| 2026-06-25 | 文档创建 | done | RGA-KD 规格 + 执行计划落定 | 写 `pilot/pilot_rga_diag.py` 跑 Phase A |
| 2026-06-25 | 环境核查 | done | **到场时仅有本文件**：`pilot/pilot_gradgeom.py`、`PILOT_RESULTS.json`、`refine-logs/EXPERIMENT_PLAN.md`、`CURRICULUM_KD_*.md` 全部不在盘上。冻结 pilot *结论*作背景沿用，*代码*在 `pilot_rga_diag.py` 中自包含重建。env=`py310_torch24`(torch2.9+cu128, 4×A100 80GB) | 写诊断脚本 |
| 2026-06-25 | A: pilot 诊断 | done | `pilot/RGA_DIAG_RESULTS.json`（GPU0 UUID-4ad1e660, 13.6s, 5seed）。r=0.30±0.11 有结构；corr(r,kdloss)=**−0.35**(非换皮)；low_r **0.356**≪random **0.896**(证伪臂强成立)；**但 high_r 在 1R 输 random、1R/2R 输 GradSpan-KD 本体(d_optimal_grad 0.942/0.953)**，仅 4R 追平 | 真实小 pair 复核 |
| 2026-06-25 | A: 决策门判定 | done | **条件 GO（偏 PIVOT-watch）**：§5.2 准则1-3 全过、无硬性 STOP；准则4 仅部分（RGA 选样未超 GradSpan-KD 幅值覆盖）；准则5 gate 在 pilot 不可测（教师 pool 100%）。→ 进真实小 pair 复核（决定性：测 gate + 验 RGA vs GradSpan 排序） | ResNet-56→20 复核 |
| 2026-06-25 | A: 真实小 pair 复核 | done | ResNet-56(70.3%)→ResNet-20/CIFAR-100, last-layer 梯度代理。**low_r 在 digits+2 真实网 regime 都可靠最差**；**high_r 在真实网所有预算 ≥ GradSpan-KD 本体（pilot 的 RGA<GradSpan 未复现，反转）**；high_r 在 20/类时为最佳法(0.242, ~3σ>random)。gate 仍未测(教师 pool 92%, corr≈0) | 写 PHASE_A_SUMMARY + 进 Phase B |
| 2026-06-25 | **A: 最终判定** | **GO** | 无硬性 STOP；准则1-3 全过，准则4 真实网达标，准则5(gate)留给 E6。定位=finding paper。主签名=studentKD+teacherENTK | Phase B 方法定稿 |
| 2026-06-25 | B: 方法定稿 | done | `RGA_KD_METHOD_FINAL.md`：签名=studentKD+teacherENTK；末层梯度代理为默认；soft gate(×p_T(y))默认+E6 定夺；D-optimal 覆盖；static 难度；预算报全曲线。`RGA_KD_PHASE_A_SUMMARY.md` 写完 | Phase C E0/E1 |
| 2026-06-25 | C: E0 有效秩 R | done | 全池 N=10000：R(energy90)=260, R(part)=929, r=0.591±0.17, corr(r,kdloss)=−0.21, CKA=0.12（含在 `E1_VISION_RESULTS.json`） | — |
| 2026-06-25 | C: E1 空间对决(GATE) | **PASS** | `pilot/E1_VISION_RESULTS.json`。**RGA 清晰胜特征空间(+6.3pt@4k,~6σ)**；RGA≥GradSpan；low_r 各预算最差；选择成本≤1.7s 可忽略。定位 finding-paper 成立 | 进 E2/E3/E4 + E6(必做) |
| 2026-06-25 | C: E2 主结果 | **PASS** | `E2_VISION_RESULTS.json`：RGA 在 b2000/b4000 胜 GRAFT&TAGCOS，b4000 为最佳法(0.429 vs DPP 0.406/GraNd 0.392/GRAFT 0.371)。算法 novelty 成立 | — |
| 2026-06-25 | C: E3 KD 相关性 | done | `E3_KDvsCE_RESULTS.json`：RGA 优势在 CE 下**更大**(+0.11 vs KD +0.07@b4000)→ **丢 KD-special 主张**，改写为 teacher-guided 选择(KD&CE 都帮) | — |
| 2026-06-25 | C: E6 gate 证伪 | **DECISIVE** | `E6_GATE_RESULTS.json`：30%污染下 gate_off 选中97.4%污染样本→acc崩到0.011；gate_on 0%污染→0.299。**gate 必不可少**（Phase A 未测项已证）| — |
| 2026-06-25 | C: E4 效率摊销 | done | `E5_ABLATION_RESULTS.json`：末层代理(0.433,0.02s)=全梯度(0.421,0.6s)质量相当但~30×更便宜；选择成本~2s/预算<<22s/学生训练，首个学生即摊销 | — |
| 2026-06-25 | C: E5 消融 | done | **覆盖关键**：D-optimal 覆盖 0.433 vs 纯 top-r 0.318(+11.5pt)；主签名最佳；对 proj-dim(512-8192)/anchors(64-1024) 稳健 | — |
| 2026-06-25 | D: 论文 | draft | `RGA_KD_PAPER.md` 全节完成（含 E5/E4 §5.4）。finding-paper 定位，诚实 limitation（预算依赖/单 pair/末层代理/教师质量） | 可选：扩 WRN/BERT pair、独立 novelty-check |
| 2026-07-01 | **PIVOT B（导师批评）** | done | 导师指出末层代理 `(p_S−p_T)⊗f` 因子分解成 token×feature、非真梯度。转真·深层参数梯度、在 Qwen 上做（复用 SaGD 基础设施） | 见下 |
| 2026-07-01 | B: 判据实验 | done | `DECISION_QWEN_RESULTS.json`：Gram CKA token↔proxy=0.87(**证实末层代理≈token 层面**)，但 token↔deep=0.31、top10%重叠=0.05(**真深层梯度是不同信号**)。B 成立 | 重训证明 |
| 2026-07-01 | B: 重训证明 | done | `RETRAIN_QWEN_RESULTS.json`（Qwen3-8B-SFT→0.6B, Dolly, 3seed, ROUGE-L）：**deep 0.246 > proxy 0.233 > token 0.221 单调**；RGA_deep(+gate+coverage) **0.2523 最佳、胜 random 0.2465**。诚实caveat：deep_topr≈random，深梯度价值靠单调序 + gate/coverage 体现，margin 温和 | 强化：更多预算/更大 eval/1.7B/SQuAD；改写论文为 LLM 版 |

---

## 12. 文件地图（在哪找什么）

| 文件 | 作用 |
|------|------|
| `RGA_KD_PROJECT.md` | **本文件**：唯一执行入口 |
| `CURRICULUM_KD_REVIEW.md` | 领域散文综述（定位、为什么这个方向） |
| `CURRICULUM_KD_DEEP_ANALYSIS.md` | 25 篇方法的机制级拆解（打分函数/调度/消融/局限） |
| `CURRICULUM_KD_SURVEY.md` | 90 篇文献清单 + 链接；本地 PDF 在 `papers/` |
| `refine-logs/EXPERIMENT_PLAN.md` | 既有 E0–E6 骨架 + GPU 协调（Phase C 复用） |
| `refine-logs/EXPERIMENT_TRACKER.md` | 每次 run 更新（含 GPU 协调日志） |
| `pilot/pilot_gradgeom.py` | **可复用基础设施**：per-sample 梯度提取、`d_optimal()`、`spectrum()`/有效秩、`kd_train()` 重训 harness |
| `pilot/PILOT_RESULTS.json` | 冻结的 pilot 证据（低秩、D-optimal 增益） |
| `papers/` | 90 篇本地 PDF（`_DOWNLOAD_LOG.tsv` 为清单） |
| `pilot/pilot_rga_diag.py` | **待写**：Phase A 诊断脚本 |
| `RGA_KD_METHOD_FINAL.md` | **待写**（Phase B）：定稿方法 |
| `RGA_KD_DIAGNOSTIC_NOTE.md` | **条件写**（触发 STOP 时）：负面诊断报告 |

---

> **起点**：从 §11 进度日志第一个 `todo` 开始——即写 `pilot/pilot_rga_diag.py` 并跑 Phase A 诊断。按 §5.2 决策门严格判 GO/STOP/PIVOT。记住：诚实的负面结果也是成功。
