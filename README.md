# 🚀 CUDA & Training Infra Learning Roadmap (12-Week Intensive)

本路线图采用任务清单模式，旨在通过 12 周的高强度学习与实践，构建大模型底层优化的核心竞争力。

---

## 📅 核心学习任务清单

### 第一阶段：CUDA 编程基础 (W1 - W4)
- [ ] **W1: 编程模型与线程管理**
    - [x] ~~理解 Grid/Block/Thread 索引映射~~
    - [x] ~~实践：编写并正确运行 `Vector Addition`~~
    - [x] ~~掌握 `nvcc` 编译流程与错误处理机制~~
- [ ] **W2: 内存模型与显存优化**
    - [ ] 掌握 Global Memory 合并访问与 Shared Memory 银行冲突 (Bank Conflict)
    - [ ] **实践：SGEMM 迭代优化** (从 Naive 版本优化至 Tiled 版本)
- [ ] **W3: 同步机制与并行规约**
    - [ ] 掌握线程块内同步 `__syncthreads()`
    - [ ] **实践：高性能 Parallel Reduction** (并行规约算法实现)
- [ ] **W4: 进阶特性 (Streams/Graphs)**
    - [ ] 掌握 `CUDA Streams` 与异步执行
    - [ ] 实践：实现计算 (Kernel) 与数据传输 (H2D/D2H) 的 Overlap

### 第二阶段：性能瓶颈分析与算子融合 (W5 - W8)
- [ ] **W5: 极限优化与 Roofline 模型**
    - [ ] 理解 Roofline 模型：判断 Compute-bound 与 Memory-bound
    - [ ] 实践：分析 SGEMM 的算术强度与显存吞吐率
- [ ] **W6: Nsight Profiling 实战**
    - [ ] 熟练使用 `Nsight Systems` (nsys) 分析时间线
    - [ ] 熟练使用 `Nsight Compute` (ncu) 分析特定算子的寄存器压力与 Cache 命中率
- [ ] **W7: 核心算子融合 (重点实践)**
    - [ ] 理解算子融合为何能减少显存读写延迟
    - [ ] **实践任务 A (CUDA)**: 实现 `RMSNorm` Warp-level 优化版 (使用 `__shfl_xor_sync`)
    - [ ] **实践任务 B (Triton vs CUDA)**: 编写 `Fused RoPE` (旋转位置编码) 算子
- [ ] **W8: Attention 机制优化**
    - [ ] 深度拆解 `FlashAttention-V2` 论文逻辑
    - [ ] **实践：编写简化版 Tile-based Attention** (理解 Tiling & Recomputation)

### 第三阶段：大模型工程化与推理技术 (W9 - W11)
- [ ] **W9: 模型转换与计算图优化**
    - [ ] 掌握 PyTorch 模型导出为 ONNX
    - [ ] 实践：编写脚本进行算子融合 (Operator Fusion) 与常量折叠 (Constant Folding)
- [ ] **W10: TensorRT 部署实践**
    - [ ] 掌握 TensorRT C++ API 构建推理 Engine
    - [ ] 实践：实现 INT8 量化校准 (Calibration) Pipeline
- [ ] **W11: vLLM 与 Triton 进阶 (重点实践)**
    - [ ] 理解 PagedAttention 如何解决 KV-Cache 内存碎片
    - [ ] **实践任务 C (PagedAttention)**: 编写基于 Block Table 的 KV-Cache 聚合 Kernel
    - [ ] **实践任务 D (Triton)**: 将 W7/W8 的算子用 Triton 重写，对比开发效率与性能

### 第四阶段：毕业项目与面试冲刺 (W12)
- [ ] **W12: 综合实战与总结**
    - [ ] **项目**: 选取一个开源小模型，使用自己的 Triton Kernel 替换原有算子
    - [ ] 进行端到端 Profiling，分析加速比
    - [ ] 复习 CUDA/C++ 八股文，准备面试冲刺

---

## 🛠️ 重点 Kernel 实践专项 (Hardcore Kernel List)

- [ ] **SGEMM Tiled Optimization** (W2): 目标是达到 cuBLAS 70% 以上性能
- [ ] **RMSNorm (Warp-level)** (W7): 目标是熟练运用 Warp Shuffle 指令
- [ ] **Fused RoPE (Vectorized)** (W7): 目标是掌握 `float4` 向量化加载技巧
- [ ] **PagedAttention Simulator** (W11): 目标是深刻理解 vLLM 显存管理机制

---

## 📚 学习资源记录
1. [ ] 阅读 《CUDA C++ Programming Guide》 (中/英)
2. [ ] 阅读 《Professional CUDA C Programming》
3. [ ] 拆解 [flash-attention](https://github.com/Dao-AILab/flash-attention) 源码
4. [ ] 拆解 [vLLM](https://github.com/vllm-project/vllm) PagedAttention 部分源码
