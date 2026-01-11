# -*- coding: utf-8 -*-
"""
======
五维超脑·天网指挥台 (Commercial Pro V12.0 - 终极全能版)

【版本特性】
1. **全能视图**: K线(日/周/月)、因子雷达、深度研报、财务数据一站式展示。
2. **多重人格**: 完美渲染 AI 的“商业拆解+宏观策略+游资博弈”多维分析。
3. **全域覆盖**: 支持全A股及各大细分板块扫描。
"""
import sys
import os
import re
from datetime import datetime, date
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
# 强制设置 Pandas 后端避免 Arrow 内存错误
os.environ["PANDAS_ARROW_NO_EXTENSION"] = "1"
# 引入核心组件时添加容错
try:
    import pandas as pd
except (ImportError, MemoryError) as e:
    pd = None
    print(f"Pandas 加载失败: {e}", file=sys.stderr)

try:
    from config_manager import get_config, update_keys, test_deepseek, test_tavily
except ImportError:
    # 降级配置管理器（无配置功能时使用内存字典）
    class MockConfig:
        def __init__(self):
            self.deepseek_api_key = ""
            self.tavily_api_key = ""
    _mock_cfg = MockConfig()
    def get_config(): return _mock_cfg
    def update_keys(**kwargs):
        for k, v in kwargs.items(): setattr(_mock_cfg, k, v)
    def test_deepseek(key): return (False, "config_manager 未实现")
    def test_tavily(key): return (False, "config_manager 未实现")

import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# 核心组件容错导入
try:
    from universe_cache import UniverseCache
except ImportError:
    class UniverseCache:
        def __init__(self): self.cache = {}
        def get(self, k): return self.cache.get(k)
        def set(self, k, v, ttl=3600): self.cache[k] = v

try:
    from data_engine import DataEngine, normalize_code, normalize_single_stock_payload, standardize_code
except ImportError:
    raise ImportError("请确保 data_engine.py 存在并实现核心方法")

try:
    from tools.connectivity_doctor_v2 import probe_endpoints
except ImportError:
    def probe_endpoints(): return {"status": "connectivity_doctor_v2 未加载"}

try:
    from market_scanner import MarketScanner, ScanConfig
except ImportError:
    class ScanConfig:
        def __init__(self, target_concepts=None): self.target_concepts = target_concepts or []
    class MarketScanner:
        def __init__(self, engine, cache): self.engine, self.cache = engine, cache
        def scan_cached(self, pool_id, config, cache_ttl=0): return (pd.DataFrame() if pd else [], None)

try:
    from deep_search_agent import DeepSearchAgent
except ImportError:
    class DeepSearchAgent:
        def __init__(self, deepseek_key, tavily_key): self.keys = (deepseek_key, tavily_key)
        def analyze_macro_situation(self): return {"core_logic": "模拟宏观分析"}

try:
    from radiation_engine import RadiationEngine
except ImportError:
    class RadiationEngine:
        def __init__(self, api_key): self.api_key = api_key
        def infer_opportunities(self, intel):
            return {"core_theme": "模拟主线", "strategy_rationale": "模拟逻辑", "target_concepts": []}

try:
    from signal_fuser import SignalFuser
except ImportError:
    class SignalFuser:
        def __init__(self, engine, cache): self.engine, self.cache = engine, cache
        def fuse_signals(self, market_df, macro_report, logic_report, hot_sentiment_stocks):
            return market_df if pd else []

try:
    from slow_factor_engine import SlowFactorEngine
except ImportError:
    class SlowFactorEngine:
        def __init__(self, cache): self.cache = cache
        def enrich_market_df(self, df, engine, macro_report, logic_report, hotlist, as_of=None, topk=150):
            return df if pd else []

try:
    from paper_portfolio import simulate_paper_portfolio
except ImportError:
    def simulate_paper_portfolio(engine, candidates, start, end, top_k, rebalance, initial_cash, stop_loss, take_profit):
        return {"ok": False, "msg": "paper_portfolio 未实现"}

try:
    from research.decision_engine import build_decision
except ImportError:
    def build_decision(dp):
        return {"decision_card": {"verdict": "WATCH", "horizon": "1m", "data_quality_score": 50, "position_sizing_pct": 0}}

try:
    import ai_advisor
except ImportError:
    class MockAIAdvisor:
        @staticmethod
        def get_ai_strategy(dp, key):
            return {"ai_score": 5.0, "decision": "观望", "setup_logic": "模拟逻辑", "scores": {}}
    ai_advisor = MockAIAdvisor()

try:
    from logging_utils import FetchResult, build_error, make_result
except ImportError:
    class FetchResult:
        def __init__(self, data, source, errors=None, fallback_used=False, cache_hit=False):
            self.data, self.source, self.errors = data, source, errors or []
            self.fallback_used, self.cache_hit = fallback_used, cache_hit
    def build_error(source, error_type, message):
        return {"source": source, "error_type": error_type, "message": message}
    def make_result(data, source, errors=None, fallback_used=False, cache_hit=False):
        return FetchResult(data, source, errors, fallback_used, cache_hit)

# Step4: backtest + weight learning (optional)
try:
    from weight_learner import learn_weights, save_weights, load_weights
    from step4_backtest import run_factor_backtest
except Exception:
    learn_weights = save_weights = load_weights = run_factor_backtest = None

# Optional pandas import to avoid pyarrow MemoryError on constrained hosts
_PD_REF: Optional[Any] = pd
_PD_ERROR: Optional[str] = None if pd else "Pandas not installed or MemoryError"

def _get_pandas():
    return _PD_REF

def _pd_available() -> bool:
    return _PD_REF is not None

def _pd_error_message() -> Optional[str]:
    return _PD_ERROR

def _safe_dataframe(df, **kwargs):
    """Render dataframe defensively to avoid Arrow conversion crashes."""
    pd = _get_pandas()
    if pd is None:
        st.write(df)
        return
    def _sanitize(frame):
        tmp = pd.DataFrame(frame).copy()
        for col in tmp.columns:
            if tmp[col].dtype == "object":
                tmp[col] = tmp[col].replace({"—": None, "--": None, "": None}).astype(str).fillna("")
            else:
                tmp[col] = pd.to_numeric(tmp[col], errors="coerce").fillna(0)
        return tmp
    try:
        return st.dataframe(_sanitize(df), **kwargs)
    except Exception as e:
        try:
            return st.dataframe(_sanitize(df).astype(str), **kwargs)
        except Exception:
            st.write("数据渲染失败，显示原始数据:")
            st.write(df.head(10))

# ==========================================
# 1. 页面全局配置
# ==========================================
st.set_page_config(
    page_title="五维超脑 | 终极商业版",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 注入顶级金融终端 CSS
st.markdown("""
<style>
    /* 战备状态颜色 */
    .defcon-attack { background-color: #d4edda; color: #155724; padding: 10px; border-left: 5px solid #28a745; }
    .defcon-defense { background-color: #fff3cd; color: #856404; padding: 10px; border-left: 5px solid #ffc107; }
    .defcon-retreat { background-color: #f8d7da; color: #721c24; padding: 10px; border-left: 5px solid #dc3545; }
    
    /* 决策高亮 */
    .decision-buy { color: #d32f2f; font-weight: 900; font-size: 1.2em; }
    .decision-sell { color: #2e7d32; font-weight: 900; font-size: 1.2em; }
    
    /* 卡片样式 */
    .metric-card { background-color: #ffffff; padding: 15px; border-radius: 8px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); margin-bottom: 10px; }
    .logic-box { background-color: #f8f9fa; border-left: 4px solid #4e8cff; padding: 15px; margin-bottom: 10px; border-radius: 4px; }
    
    /* 标签 */
    .tag-concept { background-color: #e3f2fd; color: #1565c0; padding: 2px 8px; border-radius: 12px; font-size: 0.85em; margin-right: 5px; }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. 初始化与连接
# ==========================================
st.sidebar.title("🧠 五维超脑 V12.0")
st.sidebar.caption("全能商业决策系统")

with st.sidebar.expander("🔑 神经连接 (API Keys)", expanded=True):
    cfg = get_config()
    ds_key = st.text_input("DeepSeek Key", value=(cfg.deepseek_api_key or ""), type="password", help="用于 AI 审计/宏观研判/辐射推演")
    tavily_key = st.text_input("Tavily Key (可选)", value=(cfg.tavily_api_key or ""), type="password", help="用于联网搜索（不填也可离线运行）")

    c1, c2, c3 = st.columns(3)
    if c1.button("💾 保存配置", width="stretch"):
        update_keys(deepseek_api_key=ds_key, tavily_api_key=tavily_key)
        st.success("已保存到本地配置（.w5brain_config.json）。定时指挥塔也会读取同一份配置。")

    if c2.button("🔌 测试DeepSeek", width="stretch"):
        ok, msg = test_deepseek(ds_key)
        (st.success if ok else st.error)(msg)

    if c3.button("🔌 测试Tavily", width="stretch"):
        ok, msg = test_tavily(tavily_key)
        (st.success if ok else st.warning)(msg)

    # 保存到 Session（用于本次会话立即生效）
    st.session_state["api_key"] = ds_key
    st.session_state["tavily_key"] = tavily_key

offline_mode = st.sidebar.checkbox(
    "离线样本模式（允许使用本地样本 fallback）",
    value=os.environ.get("ALLOW_OFFLINE_SAMPLES", "").lower() in {"1", "true", "yes", "on"},
    help="默认关闭。仅在无网场景下手动勾选，才会使用本地样本/占位数据。",
)
os.environ["ALLOW_OFFLINE_SAMPLES"] = "1" if offline_mode else "0"
sample_mode = st.sidebar.checkbox("样本模式(允许 sample_cache)", value=False, help="默认关闭，开启后才允许读取 sample_cache 兜底")
force_refresh = st.sidebar.checkbox("强制刷新", value=False, help="关闭缓存，强制拉取最新数据")
os.environ["ALLOW_SAMPLE_CACHE"] = "1" if sample_mode else "0"
os.environ["FORCE_REFRESH"] = "1" if force_refresh else "0"
st.session_state["offline_mode_flag"] = offline_mode
st.session_state["sample_mode"] = sample_mode
st.session_state["force_refresh"] = force_refresh

with st.sidebar.expander("🔎 连接自检", expanded=False):
    if st.button("运行 Connectivity Doctor v2", use_container_width=True):
        results = probe_endpoints()
        st.json(results)

pd_err_msg = _pd_error_message()
if pd_err_msg:
    st.sidebar.warning(
        "pandas 未加载，已切换为纯 Python 表格展示。错误: {}".format(pd_err_msg)
    )

# 初始化引擎 (单例模式)
if "init_done" not in st.session_state:
    with st.spinner("系统自检中..."):
        cache = UniverseCache()
        engine = DataEngine(cache=cache)
        st.session_state["engine"] = engine
        st.session_state["scanner"] = MarketScanner(engine=engine, cache=cache)
        st.session_state["fuser"] = SignalFuser(engine=engine, cache=cache)
        st.session_state["slow_engine"] = SlowFactorEngine(cache)
        st.session_state["init_done"] = True

# Ensure slow_engine exists even if session was created by old versions
if "slow_engine" not in st.session_state:
    try:
        _cache = getattr(st.session_state.get("engine"), "cache", None)
        st.session_state["slow_engine"] = SlowFactorEngine(_cache or UniverseCache())
    except Exception:
        st.session_state["slow_engine"] = SlowFactorEngine(UniverseCache())

# 便捷引用
ENGINE: DataEngine = st.session_state["engine"]
SCANNER: MarketScanner = st.session_state["scanner"]
FUSER: SignalFuser = st.session_state["fuser"]
SLOW: SlowFactorEngine = st.session_state.get("slow_engine")
search_agent = DeepSearchAgent(deepseek_key=ds_key, tavily_key=tavily_key)
rad_engine = RadiationEngine(api_key=ds_key)

# 模式选择
mode = st.sidebar.radio("作战模式", 
    ["☢️ 天网机会雷达 (Smart Radar)", "🔎 单股深度博弈 (Deep Game)", "📦 模拟仓 (Paper Portfolio)", "📊 策略回测 (Time Machine)"], 
    index=0
)

# ==========================================
# 2.5 兼容工具函数
# ==========================================

def _fmt_ts(ts_val: float) -> str:
    try:
        return datetime.fromtimestamp(float(ts_val)).strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        return "-"

def _fmt_val(val, suffix: str = ""):
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "—"
    if isinstance(val, (int, float)):
        return f"{val}{suffix}"
    return str(val).replace("--", "—").replace("", "—") + suffix

def _normalize_trade_date(val: Any) -> Optional[str]:
    try:
        if isinstance(val, (datetime, date)):
            return val.strftime("%Y-%m-%d")
        s = str(val).strip()
        if not s:
            return None
        if re.fullmatch(r"\d{8}", s):
            return f"{s[:4]}-{s[4:6]}-{s[6:]}"
        if re.fullmatch(r"\d{4}-\d{2}-\d{2}", s):
            return s
        return s.split(" ")[0]
    except Exception:
        return None

def _extract_latest_close(kline_obj) -> Tuple[Optional[float], Optional[str]]:
    """Safely extract the latest close and date from flexible kline payloads."""
    def _as_float(val: Any) -> Optional[float]:
        try:
            f = float(val)
            return f if not np.isnan(f) else None
        except Exception:
            return None

    close_candidates = {"close", "closing", "收盘", "收盘价"}
    date_candidates = {"date", "trade_date", "时间", "日期", "day"}
    pd = _get_pandas()

    if kline_obj is None:
        return None, None

    # pandas DataFrame path
    if pd is not None and isinstance(kline_obj, pd.DataFrame):
        df = kline_obj
        if df.empty:
            return None, None
        mapping = {str(col).lower(): col for col in df.columns}
        close_col = next((mapping[c] for c in mapping if c in close_candidates), None)
        if close_col is None:
            return None, None
        date_col = next((mapping[c] for c in mapping if c in date_candidates), None)
        df_work = df.dropna(subset=[close_col]) if close_col in df.columns else df
        if df_work.empty:
            return None, None
        if date_col and date_col in df_work.columns:
            try:
                df_work = df_work.sort_values(by=date_col)
            except Exception:
                pass
        latest_row = df_work.iloc[-1]
        return _as_float(latest_row.get(close_col)), _normalize_trade_date(latest_row.get(date_col))

    # list of dicts path
    if isinstance(kline_obj, list):
        for item in reversed(kline_obj):
            if not isinstance(item, dict):
                continue
            item_lower = {str(k).lower(): v for k, v in item.items()}
            close_val = next((item_lower.get(c) for c in close_candidates if c in item_lower), None)
            close_val = _as_float(close_val)
            if close_val is None:
                continue
            date_val = next((item_lower.get(c) for c in date_candidates if c in item_lower), None)
            return close_val, _normalize_trade_date(date_val)

    return None, None

def _render_errors(errors):
    if not errors:
        st.caption("无错误")
        return
    for err in errors:
        src = err.get("source", "?")
        typ = err.get("error_type", "?")
        msg = err.get("message", "")
        st.error(f"来源: {src} | 类型: {typ} | 详情: {msg}")

def _safe_fetch(fn, label: str) -> FetchResult:
    try:
        res = fn()
        if isinstance(res, FetchResult):
            return res
        if hasattr(res, "ok") and hasattr(res, "errors"):
            return res
        return make_result({}, source=label, errors=[build_error(label, "invalid", "返回结构非 FetchResult")])
    except Exception as e:
        return make_result({}, source=label, fallback_used=True, errors=[build_error(label, "exception", str(e))])

def _safe_holistic(engine, code: str):
    code_std = standardize_code(code)
    try:
        res = engine.single_stock(code_std)
        res = normalize_single_stock_payload(res)
        res["money_flow"] = res.get("money_flow", {})
        return res
    except Exception as e:
        return {
            "code": code_std,
            "market_data": {},
            "identity": {},
            "financial": {},
            "money_flow": {},
            "news_bundle": {},
            "diagnostics": [],
            "evidence_pack": [],
            "advice": {"action": "观望", "evidence": []},
            "_meta": {"error": str(e)},
        }

def _as_dict(val: Optional[dict]) -> dict:
    return val if isinstance(val, dict) else {}

def _errors_count(errs) -> int:
    return len(errs) if isinstance(errs, list) else 0

def _filled_from_meta(meta_obj: Optional[dict]):
    base = _as_dict(meta_obj)
    inner = _as_dict(base.get("meta"))
    return base.get("filled_metrics") or base.get("count") or inner.get("filled_metrics") or inner.get("count")

def _render_meta(meta: Optional[dict]):
    meta = _as_dict(meta)
    src = meta.get("source", "-")
    fb = "是" if meta.get("fallback_used") else "否"
    cache = "命中" if meta.get("cache_hit") else "未命中"
    ts_str = _fmt_ts(meta.get("ts"))
    filled = _filled_from_meta(meta)
    st.caption(
        "来源: {} | fallback: {} | 缓存: {} | 时间: {} | 覆盖: {}".format(
            src, fb, cache, ts_str, filled if filled is not None else "—",
        )
    )

def _minmax_norm(s):
    pd = _get_pandas()
    if pd is None or len(s) == 0:
        return [0.5] * (len(s) if hasattr(s, "__len__") else 0)
    s = pd.to_numeric(s, errors="coerce").fillna(0.0)
    mn, mx = float(s.min()), float(s.max())
    if mx - mn < 1e-9:
        return pd.Series([0.5] * len(s), index=s.index)
    return (s - mn) / (mx - mn)

def compute_entry_scores(df):
    """Compute entry_score for 'build position now' list.

    商用化闭环（Step4）：
    - 先用 Smart Radar 扫描积累慢变量
    - 使用权重学习输出 .w5brain_weights.json
    - 这里自动加载权重，把“慢变量”转成可迭代的建仓评分
    """
    pd = _get_pandas()
    if pd is None or df is None or df.empty:
        return df
    d = df.copy()
    base_col = "fused_score" if "fused_score" in d.columns else "score"
    if base_col not in d.columns:
        d[base_col] = 0.5
    base_norm = _minmax_norm(d[base_col])
    slow = pd.to_numeric(d.get("slow_score", 0.5), errors="coerce").fillna(0.5).clip(0, 1)

    learned = None
    try:
        if load_weights:
            model = load_weights(".w5brain_weights.json")
        else:
            model = None
        if isinstance(model, dict) and model.get("ok"):
            w = model.get("weights") or {}
            mu = model.get("mu") or {}
            sd = model.get("sd") or {}
            def row_score(r):
                s = 0.0
                for k, ww in w.items():
                    x = float(pd.to_numeric(r.get(k, 0.5), errors="coerce") or 0.5)
                    m = float(mu.get(k, 0.0) or 0.0)
                    ss = float(sd.get(k, 1.0) or 1.0)
                    z = (x - m) / (ss if ss != 0 else 1.0)
                    s += float(ww) * float(z)
                # sigmoid to 0~1
                return float(1.0 / (1.0 + np.exp(-2.2 * s)))
            learned = d.apply(row_score, axis=1).clip(0, 1)
            d["learned_score"] = learned.round(4)
    except Exception:
        learned = None

    if learned is not None:
        # blend: fast resonance + slow total + learned
        d["entry_score"] = (0.45 * base_norm + 0.35 * slow + 0.20 * learned).clip(0, 1).round(4)
    else:
        d["entry_score"] = (0.6 * base_norm + 0.4 * slow).clip(0, 1).round(4)

    # build action + sizing (简化版，可在 Step5+ 再迭代为更精细的风控/仓位算法)
    mp = pd.to_numeric(d.get("market_pricing", 0.5), errors="coerce").fillna(0.5).clip(0,1)
    info = pd.to_numeric(d.get("info_priced_in", 0.5), errors="coerce").fillna(0.5).clip(0,1)
    pct = pd.to_numeric(d.get("pct", 0.0), errors="coerce").fillna(0.0)
    # 追高风险：单日涨幅过大 + 量比过高
    vr = pd.to_numeric(d.get("vol_ratio", 1.0), errors="coerce").fillna(1.0)
    chase_risk = (pct >= 6.0) & (vr >= 1.8)

    def _action(row):
        es = float(row.get("entry_score", 0.0) or 0.0)
        mp_val = float(row.get("market_pricing", 0.5) or 0.5)
        info_val = float(row.get("info_priced_in", 0.5) or 0.5)
        if es >= 0.80 and mp_val >= 0.50 and info_val <= 0.72:
            return "BUILD_NOW"
        if es >= 0.68 and info_val <= 0.78:
            return "WATCH"
        return "AVOID"

    d["build_action"] = d.apply(_action, axis=1)
    d.loc[chase_risk, "build_action"] = "WATCH"

    # 仓位建议：0~12%（单票），并对拥挤度/估值做折扣
    es = pd.to_numeric(d.get("entry_score", 0.0), errors="coerce").fillna(0.0).clip(0,1)
    pos = (0.02 + es * 0.10) * (0.65 + 0.35 * mp) * (0.85 + 0.15 * (1.0 - info))
    d["position_pct"] = pos.clip(0.01, 0.12).round(4)

    # 简易止损：8%（可在模拟仓里自定义）
    d["stop_loss_pct"] = 0.08

    return d.sort_values("entry_score", ascending=False)

# ==========================================
# 3. 可视化绘图组件
# ==========================================

def plot_kline(code, freq='daily', title="K线图"):
    """绘制专业 K 线图 (日/周/月)"""
    pd = _get_pandas()
    if pd is None:
        st.warning("⚠️ Pandas 未加载，无法绘制 K 线图")
        return
    try:
        df = ENGINE.get_kline(code, freq=freq, limit=120)
    except Exception as e:
        st.warning(f"⚠️ 获取 K 线数据失败: {e}")
        return
    if df.empty:
        st.warning(f"⚠️ {title}: 暂无数据")
        return
    
    # 计算均线（容错处理）
    df['MA5'] = df['close'].rolling(5, min_periods=1).mean()
    df['MA20'] = df['close'].rolling(20, min_periods=1).mean()

    # 创建子图 (上图K线，下图成交量)
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                        vertical_spacing=0.03, row_heights=[0.7, 0.3])
    
    # K线主图
    fig.add_trace(go.Candlestick(
        x=df['date'], open=df['open'], high=df['high'], low=df['low'], close=df['close'], 
        name='K线', increasing_line_color='#eb5353', decreasing_line_color='#3bceac'
    ), row=1, col=1)
    
    # 均线
    fig.add_trace(go.Scatter(x=df['date'], y=df['MA5'], line=dict(color='orange', width=1), name='MA5'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df['date'], y=df['MA20'], line=dict(color='#4e8cff', width=1), name='MA20'), row=1, col=1)
    
    # 成交量
    colors = ['#eb5353' if r.close >= r.open else '#3bceac' for i, r in df.iterrows()]
    fig.add_trace(go.Bar(x=df['date'], y=df['volume'], marker_color=colors, name='成交量'), row=2, col=1)
    
    # 布局优化
    fig.update_layout(
        xaxis_rangeslider_visible=False, 
        height=500, 
        margin=dict(l=10, r=10, t=30, b=10), 
        template="plotly_white",
        title=dict(text=title, font=dict(size=14))
    )
    st.plotly_chart(fig, use_container_width=True)

def plot_radar(scores: dict):
    """绘制因子评分雷达图"""
    if not scores: 
        st.warning("暂无评分数据，无法绘制雷达图")
        return
    # 补全缺失维度的评分
    categories = ['宏观/行业', '基本面', '技术面', '资金面', '情绪面', '风险控制']
    values = [
        scores.get('macro_industry', 0), scores.get('fundamental', 0),
        scores.get('technical', 0), scores.get('money_flow', 0),
        scores.get('sentiment', 0), scores.get('risk_control', 0)
    ]
    # 闭合雷达
    values.append(values[0])
    categories.append(categories[0])
    
    fig = go.Figure(data=go.Scatterpolar(
        r=values, theta=categories, fill='toself', 
        line_color='#4e8cff', fillcolor='rgba(78, 140, 255, 0.3)'
    ))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 10])), 
        showlegend=False, height=300, 
        margin=dict(t=20, b=20, l=40, r=40)
    )
    st.plotly_chart(fig, use_container_width=True)

# ==========================================
# 4. 主功能逻辑
# ==========================================

# ------------------------------------------------------------------
# 模式 A: 天网机会雷达 (Smart Radar) - 全域扫描
# ------------------------------------------------------------------
if mode == "☢️ 天网机会雷达 (Smart Radar)":
    
    # --- 情报与推演 ---
    c_intel, c_logic = st.columns([1, 1])
    with c_intel:
        st.subheader("1. 全网哨兵 (Intelligence)")
        if st.button("📡 启动宏观深搜", type="primary"):
            with st.status("正在扫描全网宏观与舆情...", expanded=True):
                try:
                    st.write("抓取市场热榜...")
                    hot_spots = ENGINE.get_market_hot_spots()
                except Exception as e:
                    hot_spots = [f"热榜获取失败: {e}"]
                st.write("分析宏观定调...")
                macro_rep = search_agent.analyze_macro_situation()
                st.session_state["macro_report"] = macro_rep
                st.session_state["raw_intel"] = "\n".join(hot_spots) + "\n" + macro_rep.get("core_logic", "")
                st.success("情报搜集完成")
                st.rerun()

        if "raw_intel" in st.session_state:
            with st.expander("查看原始情报池"): st.text(st.session_state["raw_intel"])

    with c_logic:
        st.subheader("2. 辐射推演 (Radiation)")
        if st.button("🧠 激活辐射引擎", disabled="raw_intel" not in st.session_state):
            with st.spinner("AI 正在构建产业链图谱..."):
                rad_res = rad_engine.infer_opportunities(st.session_state.get("raw_intel", ""))
                st.session_state["rad_res"] = rad_res
    
    if "rad_res" in st.session_state:
        res = st.session_state["rad_res"]
        st.success(f"🎯 核心主线: {res.get('core_theme')} | 战略理由: {res.get('strategy_rationale')}")

    # --- 扫描与共振 ---
    st.markdown("---")
    st.subheader("3. 定向爆破与共振 (Blast & Fuse)")
    
    c_scan1, c_scan2 = st.columns([1, 3])
    with c_scan1:
        # 完整的全域策略池
        pool_options = {
            "🌍 全A股 (机会总览)": "all_a_shares",
            "🏆 核心资产 (Top100)": "core_assets_top100",
            "🏦 机构重仓 (Top300)": "institutional_top300",
            "🚀 创业板特攻 (30开头)": "growth_gem",
            "🐘 主板蓝筹 (60/00/002)": "sh_main_board", 
            "✨ 小市值逆向 (博弈)": "small_cap_contrarian"
        }
        sel_label = st.selectbox("🎯 选择战略底池", list(pool_options.keys()))
        pool_id = pool_options[sel_label]
        
        top_n = st.slider("显示数量", 10, 200, 50)
        
        if st.button("🚀 发射信号熔断器", type="primary"):
            target_concepts = st.session_state.get("rad_res", {}).get("target_concepts", [])
            
            with st.status("执行多维扫描...", expanded=True):
                st.write(f"正在扫描 {sel_label}...")
                config = ScanConfig(target_concepts=target_concepts)
                try:
                    df_scan, _ = SCANNER.scan_cached(pool_id, config, cache_ttl=0)
                except Exception as e:
                    st.error(f"扫描失败: {e}")
                    df_scan = pd.DataFrame() if _pd_available() else []
                
                st.write("Fuser: 执行时间轴对齐 + 宏观否决...")
                try:
                    df_final = FUSER.fuse_signals(
                        market_df=df_scan,
                        macro_report=st.session_state.get("macro_report", {}),
                        logic_report=st.session_state.get("rad_res", {}),
                        hot_sentiment_stocks=[]
                    )
                except Exception as e:
                    st.warning(f"信号融合失败，使用原始数据: {e}")
                    df_final = df_scan

                st.write("SlowFactors: computing policy/demand/substitution/pricing/info...")
                try:
                    df_final = SLOW.enrich_market_df(
                        df_final,
                        engine=ENGINE,
                        macro_report=st.session_state.get("macro_report", {}),
                        logic_report=st.session_state.get("rad_res", {}),
                        hotlist=[],
                        as_of=None,
                        topk=min(150, len(df_final) if df_final is not None else 0)
                    )
                except Exception as _e:
                    st.warning(f"慢因子计算失败: {_e}")

                try:
                    df_final = compute_entry_scores(df_final)
                    st.session_state["fused_result"] = df_final
                    st.success("扫描完成")
                except Exception as e:
                    st.error(f"评分计算失败: {e}")

    with c_scan2:
        if "fused_result" in st.session_state:
            df_res = st.session_state["fused_result"]
            pd = _get_pandas()
            if pd is not None and df_res.empty:
                st.warning("暂无符合条件的标的")
            else:
                st.success(f"🏆 最终入围: {len(df_res)} 只")
            
            # 格式化展示（动态列，避免 length mismatch）
            cols = []
            labels = []
            def add(c, l):
                if pd is not None and c in df_res.columns:
                    cols.append(c)
                    labels.append(l)
            add("code", "代码")
            add("name", "名称")
            add("sector", "行业")
            add("close", "现价")
            add("score", "量化分")
            add("fused_score", "⚡共振分")
            add("slow_score", "🐢慢变量")
            add("learned_score", "🧠权重分")
            add("fundamental_quality", "🏦财报质")
            add("fundamental_growth", "📊财报增")
            add("ops_momentum", "🛰️运营动")
            add("entry_score", "✅建仓分")
            add("slow_evidence", "证据(简)")
            if cols:
                disp = df_res[cols].head(top_n).copy()
                disp.columns = labels
                _safe_dataframe(disp, use_container_width=True)
            else:
                _safe_dataframe(df_res.head(top_n), use_container_width=True)

            # 深度审计入口
            st.divider()
            c_audit1, c_audit2 = st.columns([3, 1])
            try:
                code_list = disp["代码"].head(20).tolist() if cols else df_res["code"].head(20).tolist()
                sel_code = c_audit1.selectbox("🔍 选择标的进行深度博弈", code_list)
                if c_audit2.button("呼叫 AI 审计该股"):
                    with st.spinner("AI 正在撰写深度研报..."):
                        dp = _safe_holistic(ENGINE, sel_code)
                        dp['radiation_context'] = st.session_state.get("rad_res")
                        report = ai_advisor.get_ai_strategy(dp, ds_key)
                        st.json(report)
            except Exception as e:
                st.warning(f"审计功能异常: {e}")

# ------------------------------------------------------------------
# 模式 B: 单股深度博弈 (Deep Game) - 核心展示区
# ------------------------------------------------------------------
elif mode == "🔎 单股深度博弈 (Deep Game)":
    st.subheader("🔎 单标的显微镜 (Deep Microscope)")
    
    # 输入区
    c_in1, c_in2 = st.columns([3, 1])
    code_input = c_in1.text_input("输入代码 (如 000801)", "000801")
    
    if c_in2.button("🚀 启动博弈", type="primary"):
        if not ds_key:
            st.error("请先在左侧配置 API Key")
        else:
            with st.spinner("全息数据扫描 + AI 多重人格博弈中..."):
                dp = _safe_holistic(ENGINE, code_input)
                try:
                    report = ai_advisor.get_ai_strategy(dp, ds_key)
                    st.session_state['report'] = report
                    st.session_state['dp'] = dp
                except Exception as e:
                    st.error(f"AI 分析失败: {e}")

    # 结果展示区
    if 'report' in st.session_state and 'dp' in st.session_state:
        rep = st.session_state['report']
        dp = st.session_state['dp']
        news_bundle = _as_dict(dp.get("news_bundle"))
        meta_map = _as_dict(dp.get("_meta"))
        quote_meta = _as_dict(meta_map.get("quote"))
        identity_meta = _as_dict(meta_map.get("identity"))
        kline_meta = _as_dict(meta_map.get("kline"))
        fin_meta = _as_dict(meta_map.get("financial"))
        money_meta = _as_dict(meta_map.get("money_flow"))
        alt_meta = _as_dict(meta_map.get("alternative"))
        news_meta = _as_dict(meta_map.get("news_bundle"))
        ann_meta = _as_dict(meta_map.get("announcements"))

        # 1. 核心决策头 (Header)
        st.markdown("---")
        id_name = _as_dict(dp.get('identity')).get('name') or dp.get('code')
        sector = _as_dict(dp.get('identity')).get('sector', '') or '-'
        st.markdown(f"### {id_name} ({dp.get('code')}) | <span class='tag-concept'>{sector}</span>", unsafe_allow_html=True)

        c_h1, c_h2, c_h3, c_h4 = st.columns(4)
        md = _as_dict(dp.get('market_data'))
        raw_close = md.get('price') if md.get('price') not in (None, "", "--") else md.get('close')
        pct_delta = _fmt_val(md.get('pct') or md.get('pct_chg'), suffix="%")

        def _valid_float(val: Any) -> Optional[float]:
            try:
                f = float(val)
                return f if not np.isnan(f) else None
            except Exception:
                return None

        derived_close, derived_date = (None, None)
        close_for_display = raw_close
        if _valid_float(close_for_display) is None:
            derived_close, derived_date = _extract_latest_close(dp.get('kline_daily') or dp.get('kline'))
            if derived_close is not None:
                close_for_display = derived_close

        c_h1.metric("现价", _fmt_val(close_for_display), pct_delta)
        if derived_close is not None:
            latest_date = derived_date or "未知"
            c_h1.caption(f"非实时：使用最近收盘价 {latest_date}")
        elif quote_meta:
            source_badge = quote_meta.get('source') or '-'
            c_h1.caption(f"行情源: {source_badge}")
        c_h2.metric("AI 综合评分", _fmt_val(rep.get('ai_score')))
        st.caption(
            f"行情源: {quote_meta.get('source') or '-'} | K线源: {kline_meta.get('source') or '-'} | 财报源: {fin_meta.get('source') or '-'} | 公告源: {ann_meta.get('source') or '-'}"
        )

        # deterministic 投研结论卡
        try:
            decision_bundle = build_decision(dp)
        except Exception as e:
            st.warning(f"决策卡生成失败: {e}")
            decision_bundle = {}
        decision_card = _as_dict(decision_bundle.get("decision_card", {}))
        st.markdown("### 🧭 投研结论卡")
        c_dec1, c_dec2, c_dec3, c_dec4 = st.columns([1.4, 1, 1, 1])
        verdict = decision_card.get("verdict decision_card.get("verdict", "WATCH")
        horizon = decision_card.get("horizon", "1m")
        dq_score = decision_card.get("data_quality_score", 0)
        position_pct = decision_card.get("position_sizing_pct", 0)
        c_dec1.metric("结论", verdict)
        c_dec2.metric("持有周期", horizon)
        c_dec3.metric("数据质量", f"{dq_score}/100")
        c_dec4.metric("建议仓位", f"{position_pct}%")

        def _render_bullets(label: str, items):
            st.markdown(f"**{label}**")
            items = items or []
            if not items:
                st.caption("insufficient evidence")
                return
            for it in items:
                st.markdown(f"- {it}")

        _render_bullets("核心论据", decision_card.get("thesis", []))
        _render_bullets("关键风险", decision_card.get("risks", []))
        _render_bullets("触发条件", decision_card.get("triggers", []))
        _render_bullets("反证清单", decision_card.get("disconfirming_checklist", []))

        with st.expander("数据质量与缺口"):
            missing_notes = decision_card.get("missing_notes", []) or ["无"]
            for note in missing_notes:
                st.markdown(f"- {note}")
            ev_map = decision_card.get("evidence_map") or {}
            if ev_map:
                st.caption("证据索引")
                for key, val in ev_map.items():
                    st.text(f"{key}: {_as_dict(val).get('summary')}")

        # 决策高亮
        decision = rep.get('decision', '观望')
        color_cls = "decision-buy" if "买" in decision or "潜伏" in decision else "decision-sell" if "卖" in decision or "清" in decision else ""
        c_h3.markdown(f"**决策: <span class='{color_cls}'>{decision}</span>**", unsafe_allow_html=True)
        mc_val = md.get('market_cap')
        mc_display = _fmt_val(mc_val, suffix="亿") if mc_val not in (None, "", 0) else "—"
        c_h4.metric("市值", mc_display)

        advice = _as_dict(dp.get("advice"))
        if advice:
            st.info(f"规则建议：{advice.get('action', '观望')}")
            ev_list = advice.get("evidence") or []
            if isinstance(ev_list, list):
                for ev in ev_list:
                    ev_dict = _as_dict(ev)
                    ref = ev_dict.get("ref") or ""
                    summary = ev_dict.get("summary") or ""
                    url = ev_dict.get("url") or ""
                    prefix = f"[{ref}] " if ref else ""
                    if url:
                        st.markdown(f"- {prefix}[{summary}]({url})")
                    else:
                        st.markdown(f"- {prefix}{summary}")

        with st.expander("🩺 数据链路诊断"):
            diag_rows = []
            for key, label, meta in [
                ("quote", "行情", quote_meta),
                ("identity", "身份", identity_meta),
                ("kline", "K线", kline_meta),
                ("financial", "财务", fin_meta),
                ("money_flow", "资金", money_meta),
                ("alternative", "情报", alt_meta),
                ("news_bundle", "情报流", news_meta),
            ]:
                safe_meta = _as_dict(meta)
                diag_rows.append({
                    "module": label,
                    "source": safe_meta.get("source") or "—",
                    "fallback_used": safe_meta.get("fallback_used") or False,
                    "cache_hit": safe_meta.get("cache_hit") or False,
                    "ttl_sec": safe_meta.get("ttl_sec") or "—",
                    "retrieved_at": _fmt_ts(safe_meta.get("retrieved_at") or safe_meta.get("ts")),
                    "filled": _filled_from_meta(safe_meta) or "—",
                    "errors_count": _errors_count(safe_meta.get("errors")),
                })
            if diag_rows:
                pd = _get_pandas()
                if pd is not None:
                    _safe_dataframe(pd.DataFrame(diag_rows), use_container_width=True)
                else:
                    st.table(diag_rows)
            for label, meta in [
                ("行情", quote_meta),
                ("身份", identity_meta),
                ("K线", kline_meta),
                ("财务", fin_meta),
                ("资金", money_meta),
                ("情报", alt_meta),
                ("情报流", news_meta),
            ]:
                meta = _as_dict(meta)
                if meta.get("errors"):
                    st.markdown(f"**{label}错误详情:**")
                    _render_errors(meta.get("errors"))

        with st.expander("🐞 调试/原始数据", expanded=False):
            st.caption("single_stock payload (normalized)")
            try:
                st.json({k: v for k, v in dp.items() if k != "provider_trace"})
            except Exception as e:
                st.write(f"JSON 渲染失败: {e}")
                st.write(str(dp))
            st.caption("provider_trace")
            try:
                st.json(dp.get("provider_trace"))
            except Exception as e:
                st.write(f"provider_trace 渲染失败: {e}")

        # 2. 分页展示 (Tabs)
        tab_dash, tab_kline, tab_logic, tab_fund = st.tabs(["📊 战术看板", "📈 K线图表", "🧠 逻辑拆解", "💰 财务/资金"])

        # --- Tab 1: 战术看板 (Dashboard) ---
        with tab_dash:
            st.markdown("#### 行情/身份概览")
            col_q, col_id = st.columns(2)
            with col_q:
                price = md.get('close') or md.get('price')
                pct = md.get('pct') or md.get('pct_chg')
                if price is None:
                    derived_px, derived_dt = _extract_latest_close(dp.get('kline_daily') or dp.get('kline'))
                    price = derived_px
                    if derived_px is not None:
                        st.caption(f"(非实时：使用最近收盘价 {derived_dt or '未知'})")
                st.metric("现价", _fmt_val(price), f"{pct}%" if pct is not None else None)
                st.write(f"成交额: {_fmt_val(md.get('amount'))} | 量比: {_fmt_val(md.get('vol_ratio'))}")
                st.caption(
                    f"行情来源: {quote_meta.get('source') or '-'} | 报价: {'非实时(收盘价)' if quote_meta.get('latest_price_non_realtime') else '实时'} | 推导: {'是' if quote_meta.get('is_derived') else '否'}"
                )
                _render_meta(quote_meta)
                if not quote_meta.get("ok"):
                    _render_errors(quote_meta.get("errors"))
            with col_id:
                st.write(f"行业: {sector or '-'}")
                concepts = _as_dict(dp.get('identity')).get('concepts') or []
                st.write("概念: " + ("，".join(concepts) if concepts else "-"))
                _render_meta(identity_meta)
                if not identity_meta.get("ok"):
                    _render_errors(identity_meta.get("errors"))

            st.markdown("---")
            # 1. 核心结论 (The Setup)
            setup_logic = rep.get('setup_logic', '暂无逻辑')
            st.markdown(f"<div class='logic-box'><b>⚡ 短线逻辑 (The Setup)</b><br>{setup_logic}</div>", unsafe_allow_html=True)
            
            c_d1, c_d2 = st.columns([1, 1])
            with c_d1:
                st.markdown("#### 因子雷达评分")
                plot_radar(rep.get('scores', {}))
                
                # 风险提示
                risk_warning = rep.get('risk_warning', '暂无风险提示')
                st.warning(f"🛡️ **风险视角**: {risk_warning}")
                catalyst = rep.get('catalyst', '暂无催化剂')
                st.success(f"🔥 **强催化剂**: {catalyst}")

            with c_d2:
                # 行动计划
                plan = _as_dict(rep.get('action_plan', {}))
                st.markdown("#### 🔫 战术行动计划
