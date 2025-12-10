import os
import json
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from http import HTTPStatus

# ================= 配置区域 =================
# 在这里填入你的 Key，或者设置到环境变量中
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY") # 阿里云百炼/DashScope Key
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
# ===========================================

# 初始化库
import dashscope
from tavily import TavilyClient

dashscope.api_key = DASHSCOPE_API_KEY
tavily = None
try:
    if TAVILY_API_KEY:
        tavily = TavilyClient(api_key=TAVILY_API_KEY)
    else:
        print("⚠️ 警告：TAVILY_API_KEY 未设置，搜索功能将不可用（程序可继续运行以便调试）。")
except Exception as e:
    print("⚠️ 初始化 TavilyClient 失败：", e)
    tavily = None

app = FastAPI()

@app.get("/health")
def health():
    return {"status": "ok"}

# 允许跨域
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class TopicRequest(BaseModel):
    topic: str

# --- 核心 Agent 逻辑 ---
def get_sentiment_analysis(topic: str):
    print(f"🕵️ 正在搜索关于: {topic} ...")
    
    # 1. 使用 Tavily 搜索最新资讯
    try:
        search_result = tavily.search(query=f"{topic} 最新评论 争议 事件分析", search_depth="advanced", max_results=5)
        context = "\n".join([f"- [{res['title']}]({res['url']}): {res['content']}" for res in search_result['results']])
    except Exception as e:
        print(f"Search Error: {e}")
        context = "搜索失败，仅基于模型知识库分析。"

    print("🧠 模型正在思考...")
    
    # 2. 构建 Prompt，强制要求返回 JSON 格式以便前端渲染图表
    prompt = f"""
    你是一个高级舆情分析专家。请根据以下互联网搜索结果，对话题“{topic}”进行深度分析。
    
    搜索结果上下文：
    {context}

    请必须以严格的 JSON 格式输出，不要包含 Markdown 代码块标记（如 ```json），直接返回 JSON 字符串。
    JSON 结构要求如下：
    {{
        "sentiment_score": 0-100的整数 (0为极度负面，50中立，100极度正面),
        "sentiment_label": "正面/负面/中立/争议",
        "keywords": ["关键词1", "关键词2", "关键词3", "关键词4", "关键词5"],
        "trend_data": [
            {{"date": "最近5天的日期1", "score": 预估热度值0-100}},
            {{"date": "最近5天的日期2", "score": 预估热度值0-100}},
            ...
        ],
        "report_markdown": "这里是一篇结构清晰、排版精美的深度分析报告（Markdown格式）。请包含：事件背景、各方观点、情感分析结论、未来走势预测。请使用emoji修饰标题。"
    }}
    """

    # 3. 调用千问 (Qwen-Plus 或 Qwen-Max)
    response = dashscope.Generation.call(
        model=dashscope.Generation.Models.qwen_plus,
        prompt=prompt,
        result_format='message',  
    )

    if response.status_code == HTTPStatus.OK:
        content = response.output.choices[0].message.content
        # 简单的清洗，防止模型偶尔加 markdown 标记
        content = content.replace("```json", "").replace("```", "").strip()
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            # Fallback 如果模型没返回 JSON
            return {
                "sentiment_score": 50,
                "sentiment_label": "解析错误",
                "keywords": ["Error"],
                "trend_data": [],
                "report_markdown": f"解析模型输出失败，原始输出：\n{content}"
            }
    else:
        raise HTTPException(status_code=500, detail=f"Model Error: {response.message}")

# --- API 接口 ---
@app.post("/api/analyze")
async def analyze_sentiment(request: TopicRequest):
    data = get_sentiment_analysis(request.topic)
    return data

# --- 前端页面 (直接嵌入) ---
@app.get("/", response_class=HTMLResponse)
async def read_root():
    return """
<!DOCTYPE html>
<html lang="zh-CN" class="dark">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AI 舆情分析系统</title>
    <!-- Tailwind CSS -->
    <script src="https://cdn.tailwindcss.com"></script>
    <!-- Vue 3 -->
    <script src="https://unpkg.com/vue@3/dist/vue.global.js"></script>
    <!-- ECharts -->
    <script src="https://cdn.jsdelivr.net/npm/echarts@5.4.3/dist/echarts.min.js"></script>
    <!-- Markdown Parser -->
    <script src="https://cdn.jsdelivr.net/npm/markdown-it@13.0.2/dist/markdown-it.min.js"></script>
    
    <script>
        tailwind.config = {
            darkMode: 'class',
            theme: {
                extend: {
                    colors: {
                        primary: '#6366f1',
                        darkbg: '#0f172a',
                        cardbg: '#1e293b'
                    }
                }
            }
        }
    </script>
    <style>
        body { background-color: #0f172a; color: #e2e8f0; font-family: 'Inter', sans-serif; }
        .glass { background: rgba(30, 41, 59, 0.7); backdrop-filter: blur(10px); border: 1px solid rgba(255,255,255,0.1); }
        .markdown-body h1 { font-size: 1.5rem; font-weight: bold; margin-top: 1rem; color: #818cf8; }
        .markdown-body h2 { font-size: 1.25rem; font-weight: bold; margin-top: 1rem; color: #a5b4fc; }
        .markdown-body p { margin-bottom: 0.8rem; line-height: 1.6; color: #cbd5e1; }
        .markdown-body li { margin-left: 1.2rem; list-style-type: disc; }
        .loader { border: 4px solid #f3f3f3; border-top: 4px solid #6366f1; border-radius: 50%; width: 30px; height: 30px; animation: spin 1s linear infinite; }
        @keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }
    </style>
</head>
<body class="min-h-screen p-6">
    <div id="app" class="max-w-6xl mx-auto">
        <!-- Header -->
        <header class="mb-10 text-center">
            <h1 class="text-4xl font-extrabold text-transparent bg-clip-text bg-gradient-to-r from-indigo-400 to-cyan-400 mb-2">
                自媒体网络舆情事件分析系统
            </h1>
            <p class="text-slate-400">西藏大学-上海大学多媒体与人工智能安全研究小组</p>
        </header>

        <!-- Input Area -->
        <div class="max-w-2xl mx-auto mb-12 glass rounded-2xl p-2 flex shadow-2xl shadow-indigo-500/20">
            <input 
                v-model="topic" 
                @keyup.enter="analyze"
                type="text" 
                placeholder="输入话题，例如：'小米SU7发布会' 或 'OpenAI新模型'" 
                class="flex-1 bg-transparent border-none outline-none text-white px-4 text-lg placeholder-slate-500"
            >
            <button 
                @click="analyze" 
                :disabled="loading"
                class="bg-primary hover:bg-indigo-600 text-white px-8 py-3 rounded-xl font-medium transition-all flex items-center gap-2"
            >
                <span v-if="!loading">生成报告</span>
                <div v-else class="loader"></div>
            </button>
        </div>

        <!-- Dashboard -->
        <div v-if="result" class="grid grid-cols-1 md:grid-cols-3 gap-6 animate-fade-in-up">
            
            <!-- Left Column: Metrics -->
            <div class="space-y-6">
                <!-- Score Card -->
                <div class="glass rounded-2xl p-6 text-center relative overflow-hidden">
                    <div class="text-slate-400 text-sm uppercase tracking-wider mb-2">情感指数</div>
                    <div class="text-6xl font-bold" :class="getScoreColor(result.sentiment_score)">
                        {{ result.sentiment_score }}
                    </div>
                    <div class="text-xl mt-2 font-medium text-white">{{ result.sentiment_label }}</div>
                    <!-- Background Glow -->
                    <div class="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-32 h-32 bg-indigo-500/20 blur-3xl -z-10"></div>
                </div>

                <!-- Keywords -->
                <div class="glass rounded-2xl p-6">
                    <div class="text-slate-400 text-sm uppercase tracking-wider mb-4">舆论关键词</div>
                    <div class="flex flex-wrap gap-2">
                        <span v-for="word in result.keywords" class="px-3 py-1 bg-slate-700/50 rounded-full text-sm text-indigo-300 border border-indigo-500/30">
                            #{{ word }}
                        </span>
                    </div>
                </div>

                <!-- Trend Chart Container -->
                <div class="glass rounded-2xl p-6 h-64">
                    <div class="text-slate-400 text-sm uppercase tracking-wider mb-2">热度趋势</div>
                    <div id="trendChart" class="w-full h-full"></div>
                </div>
            </div>

            <!-- Right Column: Report -->
            <div class="md:col-span-2 glass rounded-2xl p-8 min-h-[600px]">
                <div class="flex items-center justify-between mb-6 border-b border-slate-700 pb-4">
                    <h2 class="text-2xl font-bold text-white">深度分析报告</h2>
                    <span class="text-xs text-slate-500 bg-slate-800 px-2 py-1 rounded">AI Generated</span>
                </div>
                <div class="markdown-body text-slate-300" v-html="renderedMarkdown"></div>
            </div>
        </div>
    </div>

    <script>
        const { createApp, ref, computed, nextTick } = Vue;

        createApp({
            setup() {
                const topic = ref('');
                const loading = ref(false);
                const result = ref(null);
                const mdParser = window.markdownit();

                const analyze = async () => {
                    if (!topic.value) return;
                    loading.value = true;
                    result.value = null;

                    try {
                        const res = await fetch('/api/analyze', {
                            method: 'POST',
                            headers: { 'Content-Type': 'application/json' },
                            body: JSON.stringify({ topic: topic.value })
                        });
                        const data = await res.json();
                        result.value = data;
                        
                        // Wait for DOM update then render chart
                        await nextTick();
                        initChart(data.trend_data);
                    } catch (e) {
                        alert('分析失败，请检查后端日志');
                    } finally {
                        loading.value = false;
                    }
                };

                const renderedMarkdown = computed(() => {
                    return result.value ? mdParser.render(result.value.report_markdown) : '';
                });

                const getScoreColor = (score) => {
                    if (score >= 70) return 'text-emerald-400';
                    if (score >= 40) return 'text-yellow-400';
                    return 'text-rose-400';
                };

                const initChart = (data) => {
                    if (!data || data.length === 0) return;
                    const chart = echarts.init(document.getElementById('trendChart'));
                    chart.setOption({
                        grid: { top: 10, bottom: 20, left: 30, right: 10 },
                        tooltip: { trigger: 'axis' },
                        xAxis: { 
                            type: 'category', 
                            data: data.map(i => i.date),
                            axisLine: { lineStyle: { color: '#64748b' } }
                        },
                        yAxis: { 
                            type: 'value', 
                            splitLine: { lineStyle: { color: '#334155' } },
                            axisLine: { show: false },
                            axisLabel: { color: '#64748b' }
                        },
                        series: [{
                            data: data.map(i => i.score),
                            type: 'line',
                            smooth: true,
                            lineStyle: { color: '#6366f1', width: 3 },
                            areaStyle: {
                                color: new echarts.graphic.LinearGradient(0, 0, 0, 1, [
                                    { offset: 0, color: 'rgba(99, 102, 241, 0.5)' },
                                    { offset: 1, color: 'rgba(99, 102, 241, 0)' }
                                ])
                            }
                        }]
                    });
                    window.addEventListener('resize', () => chart.resize());
                };

                return {
                    topic,
                    loading,
                    result,
                    analyze,
                    renderedMarkdown,
                    getScoreColor
                };
            }
        }).mount('#app');
    </script>
</body>
</html>
    """

if __name__ == "__main__":
    # 本地测试时可用（Render 会用 gunicorn 启动服务）
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
