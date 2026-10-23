# ActScope

**从 Tensor 出发，理解训练显存。** 一个纯浏览器运行的模块化显存工作台。

## 功能

- **Qwen3-235B-A22B、Qwen3-8B 和 Custom**：共用声明式模型与计算引擎。
- 自定义变量、Tensor shape 表达式、模块重复次数；模型定义自动保存至浏览器。
- TP-replicated、TP-sharded、EP/ETP-sharded expert 参数分类；按 Tensor 配置分片维度与粒度。
- 参数、梯度、optimizer 精度和状态分片预设；展示最重 PP stage 的每 rank 常驻显存。
- 点击 Tensor 展开本地 shape、分片组及完整计算公式。
- 激活账本：定义唯一 storage、保存者、shape 和保留 / 重算 / offload 策略。
- 模型 JSON 导入导出、响应式布局；只展示表格，不包含激活关系图。

## 本地运行

需要 Node.js 24+ 和 npm。

```sh
npm ci
npm run dev
```

打开 Vite 输出的地址。页面使用 hash 路由：`#moe`、`#dense`、`#custom`，直接刷新无需服务器回退规则。

```sh
npm test                  # 计算与图模型测试
npm run build             # 类型检查与静态构建
npm run preview           # 预览 dist
npm run test:workbench     # 需先启动服务、本机安装 Chrome
npm run test:browser       # 工作台浏览器检查
```

浏览器测试通过 `ACTSCOPE_URL` 指定服务地址。

## GitHub Pages

项目已经配置 `.github/workflows/pages.yml`：推送到 `main` 后运行测试、构建，随后部署 `dist/`。

1. 仓库必须具备 Pages 资格：公开仓库，或支持私有仓库 Pages 的 GitHub 套餐。
2. 在仓库 **Settings → Pages → Build and deployment** 中，将 Source 设为 **GitHub Actions**。
3. 推送网页源码到 `main`，或在 Actions 中手动运行 **Deploy GitHub Pages**。
4. 成功部署后的项目地址通常为 `https://<owner>.github.io/<repository>/`。

Vite 使用 `base: './'`，支持项目子路径；无 API 服务、数据库或生产环境密钥。构建产物仅包含应用和 `public/` 中的资源。

## 计算口径

模型预设只计算 decoder blocks，不含 embedding、final norm、LM head、MTP。模型结构来源：

- [Qwen3-235B-A22B config](https://huggingface.co/Qwen/Qwen3-235B-A22B/blob/main/config.json)
- [Qwen3-8B config](https://huggingface.co/Qwen/Qwen3-8B/blob/main/config.json)

默认梯度 FP32，optimizer 为 FP32 master weight + Adam m/v（共 12 bytes/parameter）；这是可编辑的训练假设，而非模型 config 定义。

```text
GPU = PP × TP × Dense DP = PP × EP × Expert TP × Expert DP

Tensor 的每 rank 常驻显存
= 本地参数量
  × (参数字节/参数分片数 + 梯度字节/梯度分片数 + optimizer 字节/optimizer 分片数)
  × 最重 stage 上的模块实例数
```

非 expert 参数使用 Dense DP；expert 使用 Expert DP。TP-replicated 表示 TP 组内复制，依然可被 DP 状态分片。分片预设为不分片、optimizer、gradient + optimizer、全部状态。

PP 展示 `ceil(模块数 / PP)` 的最重 stage。分片维度从 0 开始，expert 的第 0 维固定由 EP 切分。Attention projection 以完整 head 为分片粒度；暂不支持 KV head 复制。

**静态值不是执行峰值**：不含 all-gather、prefetch、通信、workspace 或 allocator 碎片；当前不含 CP、HSDP 和 VPP 调度。

**激活账本是用户声明的保存量，不是已验证的完整模型激活估算。** 一行代表一块唯一 storage，多个保存者写在同一行。重算与 offload 策略互斥，以避免重复扣除。预设账本留空，用户添加后才开始累计；尚未验证策略可行性、边界保存量、异步传输和生命周期峰值。`tokens_per_rank` 是本地 token 数，当前不会自动随 TP/EP 改变。

模型定义保存于本地浏览器；并行设置为会话设置。JSON 导入导出包含模型、变量、Tensor 和激活定义，不包含本次会话的并行设置。

## 开发结构

```text
src/memory/engine.ts       # 数据校验、安全表达式解析、显存计算
src/memory/presets.ts      # 模型预设与官方来源
src/memory/Workbench.tsx   # 统一模型编辑与分析页面
src/memory/workbench.css   # 工作台样式
src/memory/Tables.tsx      # 静态分项表与每 token 激活表
src/Root.tsx               # 工作台入口
```

添加模型首先扩展配置，只有新分布规则才需要改动引擎。表达式支持数字、变量、括号和 `+ - * /`，不使用 `eval`。

## License

MIT

激活表支持直接输入每 token 字节公式，例如 `hidden_size * 2`。当前 token 数由 `tokens_per_rank` 控制；表中另列 8k / 16k 的单层保存量。导入旧版 shape 格式仍受支持。
