# Agent Memory ICLR Research

Agent 记忆写入、检索、压缩与生命周期管理的研究材料和 QC-Lifecycle 实验存档。

## 目录

| 路径 | 内容 |
| --- | --- |
| `code/` | 共享工具、实验脚本、分析与验证代码 |
| `data/` | LongMemEval 数据文件 |
| `results/` | 实验输出、缓存数据与分析结果 |
| `audit/`, `verify/`, `verify_x/`, `vfy2/` | 审计和复核材料 |
| `paper/` | LaTeX 论文源文件与 PDF |
| `raw/`, `phase2_sources/` | 文献与来源资料 |
| 根目录 Markdown | 调研报告、方法设计、协议和论文草稿 |

研究入口：[`00_RESEARCH_REPORT.md`](00_RESEARCH_REPORT.md)；论文草稿：[`paper_draft.md`](paper_draft.md)。
结果文件保留各次实验的原始记录，具体指标和结论请结合相应协议及复核材料阅读。

## API 配置

代码从 `DEEPSEEK_API_KEY` 环境变量读取密钥。运行需要 API 的脚本前，在当前终端设置该变量。

PowerShell：

```powershell
$env:DEEPSEEK_API_KEY = "your-api-key"
```

Linux / macOS：

```bash
export DEEPSEEK_API_KEY="your-api-key"
```

`.env.example` 仅用于说明变量名；脚本不会自动加载 `.env`。
真实密钥、本地凭据、Python 缓存和 LaTeX 编译中间文件不进入版本控制。
