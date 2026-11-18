# ═══ 多策略 RAG 文件問答 · Streamlit + Groq + ChromaDB + LINE Bot 投放 ═══
# 執行：streamlit run app.py
from __future__ import annotations

import csv
import io
import os
import re
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import chromadb
import numpy as np
import streamlit as st
from docx import Document
from docx.oxml.table import CT_Tbl
from docx.oxml.text.paragraph import CT_P
from docx.shared import Pt, RGBColor
from docx.table import Table
from docx.text.paragraph import Paragraph
from groq import Groq
from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer

# ── 設定 ──
EMBED_MODEL = os.getenv("EMBED_MODEL", "intfloat/multilingual-e5-small")   # 可換 multilingual-e5-base、BAAI/bge-m3
MODELS = ["openai/gpt-oss-20b", "openai/gpt-oss-120b", "qwen/qwen3.8-27b"]
CHUNK, OVERLAP, CHILD, CHILD_OVERLAP, MAX_CTX = 500, 100, 200, 40, 7000
LINE_LIMIT = 500                                   # LINE 投放文字上限（字元數）
SKIP_PREFIX = ("⚠", "找不到", "請先", "（模型")      # 這些「非正式回答」不投放


def _secret(name):
    try:
        return str(st.secrets.get(name, "")).strip()
    except Exception:
        return ""


ENV_KEY = os.getenv("GROQ_API_KEY", "").strip() or _secret("GROQ_API_KEY")
ENV_LINE = os.getenv("LINE_CHANNEL_ACCESS_TOKEN", "").strip() or _secret("LINE_CHANNEL_ACCESS_TOKEN")

STRATEGIES = {
    "hybrid": ("⚡ 混合搜尋", "語意 + 關鍵字，RRF 融合排名（推薦）"),
    "semantic": ("🔍 語意搜尋", "向量相似度檢索"),
    "tfidf": ("📊 關鍵字", "字元 n-gram，適合專有名詞、條號、數字"),
    "rerank": ("🎯 重新排序", "LLM 對候選片段批次評分後重排"),
    "multi_query": ("🔄 多查詢", "LLM 改寫 3 個角度問題聯合檢索"),
    "compress": ("✂️ 壓縮", "LLM 抽取每段最相關的句子"),
    "parent_child": ("📂 父子文檔", "小片段定位，回傳完整大片段"),
    "hyde": ("💡 HyDE", "先生成假設答案再檢索"),
}
_SENT = re.compile(r"(?<=[。！？!?；;])|\n+")


def split_text(text, size, overlap):
    sents = [s.strip() for s in _SENT.split(re.sub(r"[ \t　\xa0]+", " ", text)) if s and s.strip()]
    chunks, buf, fresh = [], "", 0
    for p in (s[i:i + size] for s in sents for i in range(0, len(s), size)):
        if fresh and len(buf) + len(p) + 1 > size:
            chunks.append(buf)
            buf, fresh = (buf[-overlap:].lstrip() if overlap else ""), 0
        buf = f"{buf} {p}" if buf else p
        fresh += len(p)
    return chunks + ([buf.strip()] if fresh else [])


@dataclass
class Doc:
    sid: str
    name: str
    chunks: list
    labels: list
    n_child: int = 0
    tfidf: object = field(default=None, repr=False)
    tm: object = field(default=None, repr=False)


class LLM:
    def __init__(self, key, model):
        key = (key or "").strip()
        self.client = Groq(api_key=key) if key else None
        self.model = model or MODELS[0]

    def __call__(self, prompt, max_tokens=300, temp=0.3, system=None, raise_err=False):
        if not self.client:
            return None
        kw = dict(model=self.model, temperature=temp, max_completion_tokens=max_tokens,
                  messages=([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": prompt}])
        if "gpt-oss" in self.model:  # 推理模型：推理 token 也計入上限
            kw.update(max_completion_tokens=max(max_tokens * 3, 1024), reasoning_effort="low", include_reasoning=False)
        try:
            try:
                r = self.client.chat.completions.create(**kw)
            except Exception as e:
                if "reasoning" not in str(e).lower():
                    raise
                kw.pop("reasoning_effort", None)
                kw.pop("include_reasoning", None)
                r = self.client.chat.completions.create(**kw)
            return re.sub(r"<think>.*?</think>", "", r.choices[0].message.content or "", flags=re.S).strip()
        except Exception:
            if raise_err:
                raise
            return None


@st.cache_data(ttl=600, show_spinner=False)
def list_models(key):
    try:
        ids = [m.id for m in Groq(api_key=key.strip()).models.list().data if getattr(m, "active", True) is not False]
    except Exception:
        return []
    return sorted(i for i in ids if not any(s in i.lower() for s in ("whisper", "tts", "orpheus", "guard", "playai")))


class RAG:
    """嵌入模型 + 向量庫（全站共用）；每位使用者的文件存在各自的 session_state。"""

    def __init__(self):
        self._embedder, self._lock = None, threading.Lock()
        self.e5 = "e5" in EMBED_MODEL.lower()
        self.db = chromadb.EphemeralClient()  # 記憶體模式，重啟服務即清空
        slug = re.sub(r"[^a-zA-Z0-9]+", "_", EMBED_MODEL.split("/")[-1])[:40].strip("_")
        self.col = self.db.get_or_create_collection(f"p_{slug}", metadata={"hnsw:space": "cosine"})
        self.ccol = self.db.get_or_create_collection(f"c_{slug}", metadata={"hnsw:space": "cosine"})

    def embedder(self):
        if self._embedder is None:
            with self._lock:
                if self._embedder is None:
                    import torch
                    from sentence_transformers import SentenceTransformer
                    m = SentenceTransformer(EMBED_MODEL, device="cuda" if torch.cuda.is_available() else "cpu")
                    m.max_seq_length = min(512, m.max_seq_length or 512)
                    self._embedder = m
        return self._embedder

    # ── 文件載入（回傳 Doc，不改動全域狀態）──
    def load(self, name, data):
        ext = Path(name).suffix.lower()
        if ext not in (".pdf", ".docx"):
            raise ValueError("僅支援 PDF / DOCX")
        segs = self._pdf(data) if ext == ".pdf" else self._docx(data)
        pairs = [(c, lab) for lab, txt in segs for c in split_text(txt, CHUNK, OVERLAP)]
        if not pairs:
            raise ValueError("無可擷取文字（可能是掃描檔，需先 OCR）")
        doc = Doc(uuid.uuid4().hex, name, [p[0] for p in pairs], [p[1] for p in pairs])
        try:
            self._add(self.col, [f"{doc.sid}_{i}" for i in range(len(pairs))], doc.chunks,
                      [{"sid": doc.sid, "idx": i} for i in range(len(pairs))])
            kids = [(f"{doc.sid}_{p}_{c}", t, p) for p, par in enumerate(doc.chunks)
                    for c, t in enumerate(split_text(par, CHILD, CHILD_OVERLAP))]
            doc.n_child = len(kids)
            self._add(self.ccol, [k[0] for k in kids], [k[1] for k in kids], [{"sid": doc.sid, "idx": k[2]} for k in kids])
            doc.tfidf = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), sublinear_tf=True, max_features=50000)
            doc.tm = doc.tfidf.fit_transform(doc.chunks)
        except Exception:
            self.drop(doc)
            raise
        return doc

    def drop(self, doc):
        if doc is None:
            return
        for c in (self.col, self.ccol):
            try:
                c.delete(where={"sid": doc.sid})
            except Exception:
                pass

    def _pdf(self, data):
        return [(f"第 {i} 頁", t) for i, p in enumerate(PdfReader(io.BytesIO(data)).pages, 1) if (t := p.extract_text() or "").strip()]

    def _docx(self, data):
        doc, segs, head, buf, nt = Document(io.BytesIO(data)), [], "前言", [], 0
        flush = lambda: (segs.append((head, "\n".join(buf))), buf.clear()) if buf else None
        for el in doc.element.body.iterchildren():
            if isinstance(el, CT_P):
                para = Paragraph(el, doc)
                txt = para.text.strip()
                if not txt:
                    continue
                style = (para.style.name if para.style is not None else "") or ""
                if style.lower().startswith(("heading", "title")) or style.startswith(("標題", "标题")):
                    flush()
                    head = txt[:40]
                buf.append(txt)
            elif isinstance(el, CT_Tbl):
                nt += 1
                rows = []
                for row in Table(el, doc).rows:
                    cells = []
                    for c in row.cells:
                        t = re.sub(r"\s+", " ", c.text).strip()
                        if t and (not cells or cells[-1] != t):
                            cells.append(t)
                    if cells:
                        rows.append(" | ".join(cells))
                if rows:
                    flush()
                    segs.append((f"表格 {nt}（{head}）", "\n".join(rows)))
        flush()
        return segs

    # ── 向量與檢索 ──
    def _enc(self, texts, query=False):
        if self.e5:
            texts = [("query: " if query else "passage: ") + t for t in texts]
        return self.embedder().encode(texts, batch_size=32, normalize_embeddings=True, show_progress_bar=False).tolist()

    def _add(self, col, ids, docs, metas, b=500):
        for i in range(0, len(docs), b):
            col.add(ids=ids[i:i + b], documents=docs[i:i + b], metadatas=metas[i:i + b], embeddings=self._enc(docs[i:i + b]))

    def vec(self, doc, qs, k, child=False):
        total = doc.n_child if child else len(doc.chunks)
        if not total:
            return [[] for _ in qs]
        res = (self.ccol if child else self.col).query(query_embeddings=self._enc(qs, True), n_results=max(1, min(k, total)),
                                                       where={"sid": doc.sid}, include=["metadatas"])
        return [[int(m["idx"]) for m in ms] for ms in res["metadatas"]]

    @staticmethod
    def kw(doc, q, k):
        s = (doc.tm @ doc.tfidf.transform([q]).T).toarray().ravel()
        return [int(i) for i in np.argsort(-s)[:k] if s[i] > 0]

    @staticmethod
    def rrf(lists, k):
        sc = {}
        for lst in lists:
            for r, i in enumerate(dict.fromkeys(lst)):
                sc[i] = sc.get(i, 0) + 1 / (61 + r)
        return sorted(sc, key=sc.get, reverse=True)[:k]

    def retrieve(self, doc, q, strategy, k, llm):  # 回傳 [(片段索引, 文字)]
        hybrid = lambda n: self.rrf([self.vec(doc, [q], n * 2)[0], self.kw(doc, q, n * 2)], n)
        if strategy == "semantic":
            idx = self.vec(doc, [q], k)[0]
        elif strategy == "tfidf":
            idx = self.kw(doc, q, k)
        elif strategy == "parent_child":
            idx = list(dict.fromkeys(self.vec(doc, [q], k * 4, True)[0]))[:k]
        elif strategy == "multi_query":
            r = llm(f"將以下問題改寫成 3 個角度不同的繁體中文問題，每行一題，不加編號：\n{q}", 200, 0.7) or ""
            extra = [e for e in (l.strip("-•*·0123456789.、)） ") for l in r.splitlines()) if len(e) > 3][:3]
            idx = self.rrf(self.vec(doc, [q] + extra, k * 2), k)
        elif strategy == "hyde":
            h = llm(f"請針對以下問題寫一段約 100 字、語氣像文件原文的假設性答案（繁體中文）：\n{q}", 250, 0.7)
            idx = self.rrf(self.vec(doc, [q, h] if h else [q], k * 2), k)
        elif strategy == "rerank":
            cands = hybrid(min(max(k * 3, 6), 12))
            idx = cands[:k]
            if llm.client and len(cands) > 1:
                lst = "\n\n".join(f"[{n}] {doc.chunks[i][:400]}" for n, i in enumerate(cands, 1))
                r = llm(f"問題：{q}\n\n候選片段：\n{lst}\n\n評估每段與問題的相關度（0-10），只輸出「編號:分數」每行一個。",
                        15 * len(cands), 0) or ""
                sc = {int(a): float(b) for a, b in re.findall(r"\[?(\d+)\]?\s*[:：]\s*(\d+(?:\.\d+)?)", r)}
                idx = [cands[n] for n in sorted(range(len(cands)), key=lambda n: (-sc.get(n + 1, -1), n))[:k]]
        else:  # hybrid、compress
            idx = hybrid(k)
        hits = [(i, doc.chunks[i]) for i in idx]
        if strategy == "compress" and llm.client and hits:
            def comp(h):
                r = llm(f"從以下文本逐字擷取與問題「{q}」最相關的 1-3 句，不改寫不解釋；完全無關請輸出「無」。\n\n{h[1]}", 250, 0)
                return (h[0], r if r and r != "無" else h[1][:300])
            with ThreadPoolExecutor(min(4, len(hits))) as ex:
                hits = list(ex.map(comp, hits))
        return hits

    def answer(self, doc, q, strategy, k, model, key):
        """回傳 (回答, 參考片段, 說明列)。"""
        if doc is None:
            return "請先載入 PDF 或 DOCX 文件。", "", ""
        llm = LLM(key, model)
        t0 = time.time()
        hits = self.retrieve(doc, q, strategy, int(k), llm)
        t1 = time.time()
        blocks, used = [], 0
        for n, (i, txt) in enumerate(hits, 1):
            b = f"[片段{n}｜{doc.labels[i]}]\n{txt}"
            if used and used + len(b) > MAX_CTX:
                break
            blocks.append(b)
            used += len(b)
        ctx = "\n\n".join(blocks)
        info = f"📄 {doc.name} · {STRATEGIES[strategy][0]} · {len(blocks)} 片段 · 檢索 {t1 - t0:.2f}s"
        if not hits:
            return "找不到相關片段，請換個問法或策略。", ctx, info
        if not llm.client:
            return "⚠ 尚未設定 Groq API Key（檢索結果請看「檢索片段」）", ctx, info
        ans = llm(f"請只根據上下文回答。若上下文沒有相關資訊，請說「文件中未提及」，不要編造。\n"
                  f"每個重點後以 [片段n] 標註依據，使用繁體中文條列整理。\n\n上下文：\n{ctx}\n\n問題：{q}",
                  1024, 0.2, "你是嚴謹的文件問答助手，只依據提供的上下文作答。", True)
        return ans or "（模型未回傳內容）", ctx, f"{info} · {llm.model} · 生成 {time.time() - t1:.1f}s"


@st.cache_resource(show_spinner="⏳ 載入嵌入模型中（首次需下載，約 470 MB）…")
def get_rag():
    rag = RAG()
    rag.embedder()
    return rag


# ── 匯出：Word / CSV ──
def export_csv(records):
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=list(records[0]))
    w.writeheader()
    w.writerows(records)
    return buf.getvalue().encode("utf-8-sig")  # utf-8-sig：Excel 開啟中文不亂碼


def _md_runs(par, text):  # 將 **粗體** 轉成 Word 粗體
    for i, part in enumerate(re.split(r"\*\*(.+?)\*\*", text)):
        if part:
            par.add_run(part).bold = bool(i % 2)


def export_docx(records, with_src=True):
    doc = Document()
    st_ = doc.styles["Normal"]
    st_.font.name, st_.font.size = "Microsoft JhengHei", Pt(11)
    st_.element.rPr.rFonts.set("{http://schemas.openxmlformats.org/wordprocessingml/2006/main}eastAsia", "Microsoft JhengHei")
    doc.add_heading("RAG 文件問答紀錄", 0)
    meta = doc.add_paragraph(f"文件：{records[-1]['文件']}　·　匯出時間：{datetime.now():%Y-%m-%d %H:%M}　·　共 {len(records)} 題")
    meta.runs[0].font.color.rgb = RGBColor(0x6B, 0x72, 0x80)
    for n, r in enumerate(records, 1):
        doc.add_heading(f"Q{n}．{r['問題']}", level=1)
        info = doc.add_paragraph(f"{r['時間']}　·　{r['策略']}　·　{r['模型']}")
        info.runs[0].font.size, info.runs[0].font.color.rgb = Pt(9), RGBColor(0x6B, 0x72, 0x80)
        for line in r["回答"].splitlines():
            line = line.rstrip()
            if not line.strip():
                continue
            if m := re.match(r"^\s*#{1,6}\s+(.*)", line):
                doc.add_heading(m.group(1).replace("**", ""), level=2)
                continue
            if m := re.match(r"^(\s*)[-*•]\s+(.*)", line):
                par = doc.add_paragraph(style="List Bullet 2" if len(m.group(1)) >= 2 else "List Bullet")
                _md_runs(par, m.group(2))
            elif m := re.match(r"^\s*\d+[.)]\s+(.*)", line):
                par = doc.add_paragraph(style="List Number")
                _md_runs(par, m.group(1))
            else:
                _md_runs(doc.add_paragraph(), line)
        if with_src and r["參考片段"]:
            h = doc.add_paragraph()
            run = h.add_run("參考片段")
            run.bold, run.font.size = True, Pt(9)
            for blk in r["參考片段"].split("\n\n"):
                p = doc.add_paragraph(blk.strip())
                p.paragraph_format.left_indent = Pt(12)
                for run in p.runs:
                    run.font.size, run.font.color.rgb = Pt(8.5), RGBColor(0x55, 0x5B, 0x66)
    out = io.BytesIO()
    doc.save(out)
    return out.getvalue()


# ── LINE Bot 投放 ──
def plain(md: str) -> str:
    md = re.sub(r"\*\*(.+?)\*\*", r"\1", md)
    return re.sub(r"^\s*#{1,6}\s+", "", md, flags=re.M)


def line_text(q, ans, limit=LINE_LIMIT):
    """組成 LINE 訊息：去 Markdown 與 [片段n] 標註，總長不超過 limit 字。"""
    body = re.sub(r"\[片段\d+(?:[、,，\s]*(?:片段)?\d+)*\]", "", plain(ans))
    body = re.sub(r"[ \t]+\n", "\n", re.sub(r"\n{3,}", "\n\n", body)).strip()
    q = q if len(q) <= 60 else q[:59] + "…"
    msg = f"📚 {q}\n\n{body}"
    if len(msg) <= limit:
        return msg
    cut = msg[:limit - 1]
    nl = max(cut.rfind("\n"), cut.rfind("。"))   # 盡量在句尾或換行處截斷
    if nl > limit * 0.6:
        cut = cut[:nl + 1]
    return cut.rstrip() + "…"


def line_broadcast(token, text):
    """廣播給所有加入此官方帳號的好友（會消耗 LINE 訊息額度）。"""
    from linebot.v3.messaging import ApiClient, BroadcastRequest, Configuration, MessagingApi, TextMessage
    with ApiClient(Configuration(access_token=token.strip())) as client:
        MessagingApi(client).broadcast(BroadcastRequest(messages=[TextMessage(text=text)]))


def push_line(token, q, ans):
    if not token:
        st.warning("請先在側欄貼上 LINE Channel Access Token。")
        return
    text = line_text(q, ans)
    try:
        line_broadcast(token, text)
        st.toast(f"✓ 已投放到 LINE（{len(text)} 字）", icon="✅")
    except ImportError:
        st.error("請先安裝：pip install line-bot-sdk")
    except Exception as e:  # noqa: BLE001
        if getattr(e, "status", None) == 401:
            st.error("✗ LINE Token 無效（401）")
        else:
            st.error(f"✗ 投放失敗：{type(e).__name__}: {str(e)[:150]}")


# ═══════════════ 介面 ═══════════════
st.set_page_config(page_title="RAG 文件問答", page_icon="📚", layout="wide")
rag = get_rag()
ss = st.session_state
ss.setdefault("doc", None)
ss.setdefault("sig", None)
ss.setdefault("records", [])

with st.sidebar:
    st.title("📚 RAG 文件問答")
    st.caption("Groq × ChromaDB × LINE Bot")

    key = st.text_input("🔑 Groq API Key", value=ENV_KEY, type="password", placeholder="gsk_…").strip()
    models = list_models(key) if key else []
    if key:
        st.caption(f"✓ Key 有效 · {len(models)} 個模型" if models else "✗ Key 無效或無法取得模型")
    else:
        st.caption("⚠ 尚未設定 Key")
    opts = models or MODELS
    model = st.selectbox("模型", opts, index=opts.index(MODELS[0]) if MODELS[0] in opts else 0)

    up = st.file_uploader("文件（PDF / DOCX）", type=["pdf", "docx"])
    if up is not None and ss.sig != (up.name, up.size):
        ss.sig = (up.name, up.size)
        with st.spinner("解析並建立索引中…"):
            try:
                t0 = time.time()
                new = rag.load(up.name, up.getvalue())
                rag.drop(ss.doc)  # 新文件成功後才釋放舊文件
                ss.doc, ss.records = new, []
                st.success(f"✓ {new.name}\n\n{len(new.chunks)} 片段 · {time.time() - t0:.1f}s")
            except ValueError as e:
                st.error(f"✗ {e}")
            except Exception as e:  # noqa: BLE001
                st.error(f"✗ 載入失敗：{type(e).__name__}: {e}")
    elif ss.doc is not None:
        st.caption(f"目前文件：{ss.doc.name}（{len(ss.doc.chunks)} 片段）")

    strategy = st.selectbox("檢索策略", list(STRATEGIES), format_func=lambda s: STRATEGIES[s][0])
    st.caption(STRATEGIES[strategy][1])
    topk = st.slider("片段數量 Top-K", 1, 10, 4)

    st.divider()
    st.subheader("💬 LINE Bot 投放")
    line_token = st.text_input("Channel Access Token", value=ENV_LINE, type="password",
                               placeholder="貼上 LINE Channel Access Token").strip()
    line_auto = st.checkbox(f"回答後自動投放（{LINE_LIMIT} 字內）")
    st.caption("廣播給所有好友並消耗訊息額度，請確認內容。")

    st.divider()
    st.subheader("匯出問答紀錄")
    with_src = st.checkbox("Word 包含參考片段", value=True)
    export_box = st.container()   # 按鈕在腳本結尾才建立，確保包含最新一題


def render(i, r):
    with st.chat_message("user"):
        st.write(r["問題"])
    with st.chat_message("assistant"):
        st.markdown(r["回答"])
        st.caption(r["說明"])
        with st.expander("🔎 檢索片段"):
            st.text(r["參考片段"] or "（無）")
        if not r["回答"].startswith(SKIP_PREFIX):
            if st.button("📤 投放到 LINE", key=f"line_{i}"):
                push_line(line_token, r["問題"], r["回答"])


st.header("多策略 RAG 文件問答")
if ss.doc is None:
    st.info("請先在左側上傳 PDF 或 DOCX，再開始提問。例如：這份文件的重點是什麼？")

for i, r in enumerate(ss.records):
    render(i, r)

if q := st.chat_input("輸入問題，Enter 送出"):
    if ss.doc is None:
        st.warning("請先選擇 PDF 或 DOCX 文件。")
    else:
        with st.chat_message("user"):
            st.write(q)
        with st.chat_message("assistant"), st.spinner("思考中…"):
            try:
                ans, ctx, info = rag.answer(ss.doc, q, strategy, topk, model, key)
                err = None
            except Exception as e:  # noqa: BLE001
                ans, ctx, info, err = f"✗ {type(e).__name__}: {e}", "", "", e
        if err is None:
            ss.records.append({"時間": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "文件": ss.doc.name, "問題": q,
                               "回答": ans, "策略": STRATEGIES[strategy][0], "模型": model,
                               "參考片段": ctx, "說明": info})
            render(len(ss.records) - 1, ss.records[-1])
            if line_auto and line_token and not ans.startswith(SKIP_PREFIX):
                push_line(line_token, q, ans)
        else:
            st.error(ans)

with export_box:
    recs = ss.records
    c1, c2 = st.columns(2)
    c1.download_button("📄 Word", data=export_docx(recs, with_src) if recs else b"", disabled=not recs,
                       file_name=f"RAG問答紀錄_{datetime.now():%Y%m%d_%H%M%S}.docx",
                       mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                       use_container_width=True)
    c2.download_button("📊 CSV", data=export_csv(recs) if recs else b"", disabled=not recs,
                       file_name=f"RAG問答紀錄_{datetime.now():%Y%m%d_%H%M%S}.csv", mime="text/csv",
                       use_container_width=True)
    if st.button("🗑 清除對話", use_container_width=True):
        ss.records = []
        st.rerun()
