# Competitor Intel Harvester

输入竞品名、域名，或只输入我方产品信息后，自动完成：

1. 先分析我方产品定位，生成品类化采集计划。
2. 如果没有输入竞品，先自动发现候选竞品，并保留发现搜索词、来源 URL、官方入口和入池理由。
3. 在抓取前制定竞品发现策略和数据源策略，区分官网、垂直来源、论坛社区、App/社媒/视频和海量搜索。
4. 用 SearXNG 搜索竞品官网、功能页、定价页、新闻、图片结果。
5. 用 Crawl4AI 抓取重点页面，提取 Markdown、链接、图片 URL 和 PM 字段。
6. 先下载 SearXNG 图片结果，再用 icrawler 按关键词补充产品图、截图、Logo 等图片。
7. 对反爬、低文本、视频和社媒候选做公开快照复核；普通页面保存截图/文本，视频优先用 `yt-dlp` 保存公开元数据和时间点线索。
8. 抽取结构化事实并做事实聚类，例如价格、重量、尺码、材质、颜色、认证、API、额度、安全等。
9. 加载本地训练模型，给候选来源补充收录/排除/待核实分数。
10. 调用本地 Codex CLI 做 AI 收录判断和竞品分析。
11. 导出最终分析报告、收录决策、图片索引、证据审计和人工抽样标注表；调试文件保留在任务目录内。

## 安装

建议用虚拟环境：

```bash
git clone https://github.com/William-khan/-.git competitor-intel-harvester
cd competitor-intel-harvester
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
crawl4ai-setup
python -m playwright install chromium
```

如果你已经全局部署好了 `crawl4ai`、`icrawler`，也可以直接用对应 Python 环境运行。`yt-dlp` 用于视频和部分社媒公开元数据抓取，只读取公开页面元数据，不下载视频正文。

安装为可编辑包后，会得到三个命令：

```bash
competitor-intel-ui
competitor-intel-harvester
competitor-intel-train
```

`competitor-intel-ui` 启动本地页面，`competitor-intel-harvester` 跑完整命令行采集，`competitor-intel-train` 用人工标注训练本地筛选模型并生成权重文件。

## 准备 SearXNG

如果已经有本地 SearXNG 源码和虚拟环境，可以设置 `SEARXNG_DIR` 后启动：

```bash
SEARXNG_DIR=/path/to/searxng ./start_local_searxng.sh
```

默认地址：

```text
http://localhost:8888
```

确认 SearXNG 可以返回 JSON：

```bash
curl 'http://localhost:8888/search?q=notion%20pricing&format=json'
```

如果你的 SearXNG 地址不是 `http://localhost:8888`，运行时传 `--searxng-url`，或设置：

```bash
export SEARXNG_URL='http://你的-searxng地址'
```

如果你有本地代理，例如 `7897`，它通常不是 SearXNG 本身，而是给 HTTP 请求出网用的代理端口。代理要填到 `--proxy-url` 或页面里的「代理地址」：

```bash
python competitor_harvester.py "Gamma" \
  --searxng-url http://你的-searxng地址 \
  --proxy-url http://127.0.0.1:7897
```

简单判断方式：

```bash
curl 'http://127.0.0.1:7897/search?q=test&format=json'
curl -x 'http://127.0.0.1:7897' 'http://example.com'
```

如果第一个不是 JSON、第二个能打开网页，说明 `7897` 是代理，不是 SearXNG API。

## 使用

### 图形界面

启动本地 UI：

```bash
competitor-intel-ui
```

打开：

```text
http://127.0.0.1:8765
```

在页面里填竞品名单、SearXNG 地址和采集参数；如果希望报告给出对自己产品的方向建议，可以补充「我方产品名称」「我方产品定位」「我方补充背景」。点击「开始采集」后，产物会保存在：

```text
runs/<任务ID>/
```

页面会显示关键进度，并提供轻量交付物下载：`竞品分析报告_图片内嵌版.md`、`问题页面核验清单.md/csv`、`抓取前采集计划.md`、`采集原则和筛选原则.md`、`所有采集来源.csv`、`未经筛选的采集内容.md`、`筛选后的采集内容.md`、`结构化事实.csv`、`事实聚类.md/csv`、`人工抽样标注表.csv`。正式分析报告只保留图片内嵌版，避免相对图片路径失效。反爬、403、正文不足、登录超时、视频缺时间点、待核实来源会合并到 `问题页面核验清单.md/csv`；原来的分散队列仍作为内部兼容文件留在任务目录，不作为默认交付物展示。

任务结束后，如果存在需要核验的问题页面，UI 会弹出人工核验面板，告诉你每类页面需要确认什么、如何填写 `human_label`。填完 `问题页面核验清单.csv` 或 `人工抽样标注表.csv` 后，点击“训练筛选模型”，程序会把核验结果反馈到本地 `.pt` 模型和搜索卡片。

### 可选 Dify DSL

Dify 不是这个工具的必需依赖。本地 UI 已经能完成采集、筛选、Codex 分析、训练和导出；命令行也能直接跑完整流程。Dify 只适合在你想把它嵌入公司已有工作流、表单、审批或机器人入口时使用。

Dify 导入文件：

```text
competitor_intel_harvester.dify.dsl.yml
```

导入后，需要在两个 LLM 节点里配置大模型：

- `生成新任务回复`
- `生成查询结果回复`

如果 Dify 和采集器都在本机直接运行，`本地采集器地址` 填：

```text
http://127.0.0.1:8765
```

如果 Dify 在 Docker 或另一台服务器里，`127.0.0.1` 指的是 Dify 自己，不是你的 Mac。此时先让采集器监听外部访问：

```bash
python3 app.py --host 0.0.0.0 --port 8765
```

再把 Dify 里的 `本地采集器地址` 改成 Dify 后端能访问到的宿主机地址，例如：

```text
http://192.168.x.x:8765
```

Mac 上可用下面命令查看当前 Wi-Fi 局域网 IP：

```bash
ipconfig getifaddr en0
```

### 命令行

最小例子：

```bash
python competitor_harvester.py "Gamma" "Beautiful.ai" "Canva AI" \
  --searxng-url http://localhost:8888 \
  --own-product-name "竞品情报采集器" \
  --own-product-positioning "面向产品经理的一站式竞品情报采集、筛选与分析工具" \
  --out ./runs/ai-ppt
```

只填我方产品、让系统先自动发现竞品：

```bash
python competitor_harvester.py \
  --own-product-name "AI 演示文稿工具" \
  --own-product-positioning "帮助产品经理快速生成汇报和网页演示" \
  --searxng-url http://localhost:8888 \
  --max-discovered-competitors 6 \
  --out ./runs/auto-discovery
```

从文件读取竞品列表：

```bash
python competitor_harvester.py --file competitors.txt --out ./runs/my-market
```

偏国内图片：

```bash
python competitor_harvester.py "飞书" "钉钉" "企业微信" \
  --image-engine baidu \
  --out ./runs/cn-collab
```

只搜索和导出 URL，不抓网页/下载图片：

```bash
python competitor_harvester.py "Gamma" --skip-crawl --skip-images
```

## 输出

根目录默认只保留轻量交付物：

- `竞品分析报告_图片内嵌版.md`：唯一正式报告，图片已用 base64 内嵌，单独打开也不会丢图。
- `问题页面核验清单.md` / `问题页面核验清单.csv`：统一收纳所有有问题页面，包含反爬、403、Cloudflare、验证码、正文不足、登录超时、视频缺时间点、待核实来源等，并标明需要核验什么、建议如何入库。
- `所有采集来源.csv`：所有搜索、抓取、图片和下载来源，保留可追溯线索。
- `未经筛选的采集内容.md`：原始候选摘要、抓取摘要和图片候选。
- `筛选后的采集内容.md`：进入 Codex 分析前的证据池。
- `抓取前采集计划.md`：根据我方产品、竞品和品类生成的搜索词、排除词、字段、数据源策略和价值判断规则。
- `采集原则和筛选原则.md` / `收录过滤策略设计.md`：本轮采集、筛选、证据追溯、去重和入库原则。
- `结构化事实.csv`、`事实聚类.md` / `事实聚类.csv`：价格、材质、重量、尺码、颜色、认证、API、额度、安全等事实，以及同一事实的主证据/补充证据聚合。
- `自动竞品发现.md` / `自动竞品发现.csv`：无竞品输入时生成；有明确竞品输入时可能为空。
- `人工抽样标注表.csv`：用于给本地筛选模型追加训练样本。
- `本地筛选模型状态.json`：本轮是否加载了本地 `.pt` 筛选模型。

旧版英文文件、原始 JSON、Codex 输入输出、分散的登录/GUI/人工复核队列和运行日志会归档到 `_internal/`。图片下载目录和 GUI 快照目录保留在根目录，保证 CSV 里的本地路径仍可追溯。这些内部文件仍可用于排查和二次处理，但不会默认堆在根目录里。正式报告只保留 `竞品分析报告_图片内嵌版.md`，不再额外暴露无内嵌图片的报告副本。

CSV 文件使用 `utf-8-sig` 写入，Excel/Numbers 直接打开中文不应再乱码。

## 分析模板

`analysis_dimensions/*.yml` 用来沉淀成熟竞品分析的维度脚本。程序会在抓取前根据我方产品、竞品名和品类自动匹配模板，把模板里的分析维度、必找证据、来源优先级和报告结构合并进采集计划。当前已内置 `autonomous_vehicle_robotaxi` 模板，适合无人车、Robotaxi、自动驾驶整车类竞品分析。

模板只保存可复用的方法，不保存内部原文、私有链接或账号内容。开源使用者可以继续新增自己的品类模板；每次人工核验后生成的搜索卡片会和模板一起影响下一轮检索与筛选。

## 本地训练闭环

训练目标不是微调 Codex、Claude Code、Cursor 或 DeepSeek，而是训练本工具自己的本地筛选模型。它学习人工复核结果，下一轮给候选来源生成：

```text
ml_include_score
ml_exclude_score
ml_verify_later_score
ml_label
ml_adjustment
```

推荐流程：

1. 跑一次采集任务。
2. 打开 `问题页面核验清单.csv` 或 `人工抽样标注表.csv`。
3. 给样本填写 `human_label`：`include`、`exclude` 或 `verify_later`。
4. 填写 `human_reason`，说明为什么这样判断。
5. 把这些行追加到 `training_data/review_labels.csv`。
6. 保留 `product_category`、`product_type_key`、`product_type_label` 和 `search_card_candidate` 这些列。它们决定这条核验经验应该沉淀到哪一类产品的搜索卡片里。
7. 在 UI 里点击“训练筛选模型”，或运行：

```bash
python3 train_filter_model.py \
  --labels training_data/review_labels.csv \
  --model-out models/filter_model.pt \
  --cards-dir search_cards
```

训练完成后会生成 `models/filter_model.pt`、`models/本地筛选模型.pt`、`models/filter_weights.json` 和 `models/本地筛选模型权重.json`。下一次采集会自动加载 `models/filter_model.pt` 和 `search_cards/` 中匹配本轮品类的搜索卡片。模型影响“这条来源要不要收录/排除/待核实”，权重文件解释模型为什么这样判断，搜索卡片影响“下一轮同类产品要多搜哪些词、优先看哪些可复用来源、提前排除哪些常见噪声”。登录页、购物车、验证码、私有接口、付费墙、破解下载等硬规则仍然由规则层拦截，不会被模型或卡片覆盖。

本地训练层是独立于大模型供应商的。使用者可以继续用 Codex、Claude Code、Cursor、DeepSeek 或其他 OpenAI-compatible 本地模型做报告生成；`filter_model.pt` / `filter_weights.json` 只改善本工具自己的证据筛选策略，人工标注越多，边界样本越少需要人介入。

搜索卡片是增量生成的，不需要提前穷举所有产品类型。默认同一产品类型有 3 条人工标注就会先生成低置信卡片；样本变多后，卡片置信度会随之提高。比如做过一轮滑雪头盔并完成核验后，会生成 `search_cards/snow_helmet.json`；下一次再测同类产品时，系统会把历史核验过的尺码、重量、材质、认证、视频复核、行业测评站等经验合并进抓取前计划。没有产品类型字段的老样本只训练筛选模型，不生成搜索卡片，避免把不同品类经验混在一起。后续版本可以随项目一起发布几张成熟卡片，但本地用户也能继续训练自己的卡片库。

如果要指定模型路径：

```bash
python3 competitor_harvester.py "Gamma" \
  --ml-model models/filter_model.pt
```

如果只想用规则和 Codex，不用本地模型：

```bash
python3 competitor_harvester.py "Gamma" --disable-ml-filter
```

如果只想临时不用搜索卡片：

```bash
python3 competitor_harvester.py "Gamma" --disable-search-cards
```

## 筛选逻辑

搜索阶段会尽量保留所有来源；进入 Crawl4AI 的页面会更严格。程序会优先识别官网域名，抽取搜索摘要里出现的官网直达链接。若搜索没有发现官网，会先尝试 `{brand}.com`、`{brand}.app`、`{brand}.ai` 等官网探测，再补抓 `pricing`、`features`、`product`、`customers`、`docs`、`security`、`changelog` 等核心路径。

抓取前会先生成两类策略。第一类是竞品发现策略：对用户输入的竞品做官网和别名核验；如果竞品不完整，则从我方产品定位里抽取用户任务、目标人群、核心能力和购买场景，去找直接竞品、相邻竞品、替代方案和待核实候选。第二类是数据源策略：官网和官方文档优先做主证据，垂直品类站和应用商店做验证，论坛社区做口碑和痛点线索，App/社媒/视频保留公开 URL、时间点和截图，海量搜索只做兜底发现。每个来源都要能追溯到搜索词、URL、页面标题、抓取时间、平台、视频时间点或截图。

除最后的海量搜索兜底外，官网、论坛、App、社媒、视频和垂直网站也默认先走 SearXNG 定向搜索，例如 `竞品名 + site:youtube.com demo`、`竞品名 + site:reddit.com review`、`竞品名 + site:producthunt.com`。系统先找到具体入口，再决定自动抓取、保留为线索，或进入 GUI 复核。

站点适配器会优先处理这些公开来源：YouTube/Bilibili/TikTok/抖音等视频页会抓公开元数据和时间点线索；App Store、Google Play、Chrome Web Store 会抓版本、评分、截图和描述；GitHub 会抓仓库描述、Star、License、更新时间；Product Hunt、Reddit、知乎、V2EX 等会保存公开页面元数据和正文快照。适配器失败时，不会尝试破解或越权，会把来源留在 GUI/待核实队列。

结构化抽取采用“schema 先行，规则兜底”：抓取前根据品类计划确定字段，例如实物产品看参数、材质、重量、尺码、颜色、认证，AI/软件看 API、SDK、Webhook、集成、额度、安全、部署。每条结构化事实都会带 `value`、`normalized_value`、`evidence_text`、`source_url`、`source_title`、`extraction_method` 和 `needs_verification`，方便人工验收和后续训练。

证据审计会输出价值判断字段：

- `value_signals`：命中了哪些价值原则，例如竞品绑定、决策相关、信息增量、来源可信、可追溯、可获取。
- `value_missing`：缺了哪些价值原则。
- `value_verdict`：该来源是可抓/可复核、低信心线索，还是低价值噪声。
- `gui_review_candidate` / `gui_review_value_reason`：是否需要 GUI 复核，以及为什么值得复核。

`--max-pages` 是每个竞品的抓取上限，不是必须抓满的目标。有官网候选时，目录站、SEO 聚合页、登录页、论坛/社媒、反爬评论站和同名无关产品会留在 `all_sources.csv` / `evidence_audit.csv` 里做审计，但不会送进 Crawl4AI 污染正文。第三方页面只保留少量高价值验证源，例如客户案例、发布稿或可信公开报道。

反爬页面不会直接进入最终事实结论。若它看起来是官方定价、功能、文档、安全、客户案例等核心页，会进入 `问题页面核验清单.md/csv`。登录/注册/账号权限页会先进入 `需登录队列.md/csv`，按竞品和域名去重；公开页面继续采集。UI 中默认开启登录辅助后，程序复用同一个可见浏览器登录态，网页抓取结束后最多再等待 120 秒：用户已登录且页面变为可读时，会保存文本快照和截图，仍不可读则标记为“超时未人工登录”或“需账号权限”。不做验证码破解、登录/付费/访问控制绕过，不保存账号凭据，也不调用未授权私有接口。

最终报告按 Productboard + Asana 综合框架输出：

0. 核心结论与决策建议
1. 采集范围、来源与证据等级
2. 竞品概览与定位
3. 产品能力与工作流对比
4. 定价、套餐与商业化包装
5. 目标用户、市场与 GTM
6. 客户体验、服务支持与产品质量
7. 视觉与产品证据
8. 牵引、更新节奏与战略方向
9. SWOT 与风险机会
10. 横向对比矩阵
11. 信息缺口与问题页面核验
12. 建议下一步与监控计划
13. 我方产品方向分析

## 常用参数

```text
--per-query 8              每个搜索词返回多少条
--max-pages 8              每个竞品最多抓多少页
--crawl-concurrency 3      Crawl4AI 并发数
--max-discovered-competitors 6  无竞品输入时最多自动发现几个候选竞品
--proxy-url URL            可选代理地址，例如 http://127.0.0.1:7897
--image-engine bing        icrawler 引擎：bing / baidu / google
--max-image-downloads 20   每个竞品下载多少张图片
--no-cn                    不追加中文搜索词
--codex-review             调用本地 Codex CLI 做 AI 收录分析
--require-codex-review     Codex 分析失败时任务失败，不退回规则版报告
--codex-model MODEL        可选，指定 Codex 模型
--skip-gui-review          只输出复核队列，不自动做公开快照
--login-assist             登录/注册页进入集中队列；用户点击 UI 登录池链接后才打开并复用同一浏览器登录态保存快照
--login-assist-wait 120    公开网页抓取结束后的登录等待秒数
--ml-model PATH            本地训练筛选模型路径，默认 models/filter_model.pt
--disable-ml-filter        不加载本地训练模型
--ml-auto-include-threshold 0.75  模型自动提升为 accepted 的收录分阈值
--ml-auto-exclude-threshold 0.80  模型自动降级为 rejected 的排除分阈值
--search-cards-dir PATH    历史搜索卡片目录，默认 search_cards/
--disable-search-cards     不加载历史搜索卡片
```

图片下载顺序：如果 SearXNG 已启用 `images` 分类，工具会先把图片搜索结果下载到 `downloaded_images/<竞品>/searxng/`；如果 SearXNG 没开图片分类，会在日志中提示并继续用 icrawler 关键词图片搜索兜底。

## 工作流建议

第一轮跑宽一点：

```bash
python competitor_harvester.py --file competitors.txt --per-query 12 --max-pages 10 --out ./runs/baseline
```

然后人工看 `所有采集来源.csv`、`结构化事实.csv` 和 `问题页面核验清单.csv`，把真正有价值的官网、定价页、功能页沉淀成监控清单，再交给 ScopeHound 做持续变化监控。
