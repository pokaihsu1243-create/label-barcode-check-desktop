# -*- coding: utf-8 -*-
"""
標籤條碼檢查（網頁版）
啟動後自動開瀏覽器：把標籤稿 PDF／照片拖進去 → 逐個條碼解碼並與下方文字比對 → 出結果。
全程離線，不連網。檢查邏輯完全共用 check_labels.py。
"""
import base64
import io
import json
import os
import shutil
import sys
import tempfile
import threading
import time
import traceback
import urllib.parse
import uuid
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
UPLOAD_DIR = os.path.join(HERE, "_uploads")
PORT = 8720
CHECK_LOCK = threading.Lock()

import check_labels as C     # noqa: E402


def warmup():
    try:
        C.ocr_engine()
    except Exception:
        traceback.print_exc()


def run_check(path, expect):
    # OCR and template caches are shared by request threads. Keep each file's work together.
    with CHECK_LOCK:
        rows, warns, counted = C.check_pdf(path, expect)
    ng = [r for r in rows if r["verdict"] == "NG"]
    chk = [r for r in rows if r["verdict"] in ("CHECK", "NOTEXT", "FORMAT")]
    headline, tone, _n, _c = C.summarize(rows, warns, counted)
    ng_lines, seen = [], set()
    for r in ng:
        key = (r["bc"], r["ocr"])
        if key in seen:
            continue
        seen.add(key)
        ng_lines.append("第%d頁 第%s張標籤：條碼掃出 %s ，但下方文字印 %s"
                        % (r["page"], r["label"] if r["label"] > 0 else "?", r["bc"], r["ocr"]))
    return dict(
        name=os.path.basename(path), headline=headline, tone=tone,
        total=len(rows), ok=len(rows) - len(ng) - len(chk), ng=len(ng), check=len(chk),
        warns=warns, counted=counted, ng_lines=ng_lines,
        codes=sorted({r["bc"] for r in rows}),
        rows=[{k: v for k, v in r.items() if k != "glyphs"} for r in rows],
        report_html=C.build_html(path, rows, warns, counted),
    )


PAGE = r"""<!doctype html><html lang="zh-Hant"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>標籤條碼檢查</title><style>
*{box-sizing:border-box}
body{font-family:"Microsoft JhengHei","Segoe UI",sans-serif;margin:0;background:#f4f6f8;color:#1f2933;line-height:1.6}
header{background:#1f3a5f;color:#fff;padding:18px 28px}
header h1{margin:0;font-size:20px} header p{margin:2px 0 0;opacity:.85;font-size:13px}
.wrap{max-width:1600px;margin:0 auto;padding:20px 24px 60px}
#drop{border:3px dashed #b6c2d1;border-radius:14px;background:#fff;padding:40px 20px;text-align:center;
      cursor:pointer;transition:.15s}
#drop.hot{border-color:#1f3a5f;background:#eef4ff}
#drop .big{font-size:17px;font-weight:700;margin-bottom:4px}
#drop .sm{font-size:13px;color:#64748b}
.opts{display:flex;gap:14px;align-items:center;flex-wrap:wrap;margin:14px 0 0;font-size:13px;color:#475569}
.opts input[type=number]{width:90px;padding:6px 8px;border:1px solid #cbd5e1;border-radius:6px;font-size:14px}
.opts input[type=text]{flex:1;min-width:240px;padding:6px 10px;border:1px solid #cbd5e1;border-radius:6px;font-size:13px}
button{background:#1f3a5f;color:#fff;border:0;border-radius:8px;padding:8px 16px;font-size:14px;cursor:pointer}
button.ghost{background:#fff;color:#1f3a5f;border:1px solid #b6c2d1}
button:disabled{opacity:.5;cursor:default}
#status{margin:18px 0 0;font-size:14px;color:#475569;min-height:22px}
.spin{display:inline-block;width:13px;height:13px;border:2px solid #cbd5e1;border-top-color:#1f3a5f;
      border-radius:50%;animation:s .7s linear infinite;vertical-align:-2px;margin-right:6px}
@keyframes s{to{transform:rotate(360deg)}}
.file{background:#fff;border:1px solid #e2e8f0;border-radius:12px;margin:18px 0;overflow:hidden}
.fhead{padding:14px 18px;border-bottom:1px solid #eef2f6;display:flex;gap:12px;align-items:center;flex-wrap:wrap}
.fname{font-weight:700;font-size:15px;flex:1;word-break:break-all}
.banner{padding:14px 18px;font-size:19px;font-weight:700}
.ok{background:#f0fdf4;color:#0a7d32} .bad{background:#fff1f0;color:#c00} .warn{background:#fffbe6;color:#b26a00}
.stats{display:flex;gap:10px;padding:12px 18px;flex-wrap:wrap}
.stat{background:#f8fafc;border:1px solid #e2e8f0;border-radius:8px;padding:8px 14px;font-size:13px;min-width:92px}
.stat b{display:block;font-size:22px;line-height:1.2}
.s-ok b{color:#0a7d32} .s-ng b{color:#c00} .s-ck b{color:#b26a00}
.ngbox{margin:0 18px 14px;background:#fff1f0;border:1px solid #ffa39e;border-radius:8px;padding:12px 14px}
.ngbox h4{margin:0 0 6px;font-size:13px}
.ngbox pre{margin:0;font-family:Consolas,monospace;font-size:13px;white-space:pre-wrap}
.warns{margin:0 18px 14px;background:#fff7e6;border:1px solid #ffd591;border-radius:8px;padding:10px 14px 10px 30px;font-size:13px}
.warns li{margin:2px 0}
.tw{overflow-x:auto}
table{border-collapse:collapse;width:100%;min-width:980px}
th,td{border-top:1px solid #eef2f6;padding:10px;font-size:13px;vertical-align:middle;text-align:left}
th{background:#f8fafc;font-size:12px;color:#64748b}
td img{max-width:240px;display:block;border:1px solid #eef2f6;border-radius:4px}
td img.cmp{max-width:none;width:430px}
.tag{display:inline-block;width:34px;font-size:10.5px;color:#fff;border-radius:4px;text-align:center;
     margin-right:6px;vertical-align:1px}
.t-bc{background:#d60000} .t-ocr{background:#0052cc} .t-sh{background:#64748b}
.tplbad{color:#c00;font-weight:700;font-size:12px;margin-top:3px}
.tplok{color:#0a7d32;font-size:11px;margin-top:3px}
.tplweak{color:#b26a00;font-size:11.5px;margin-top:3px}
.ovwrap{margin:0 18px 16px} .ovwrap h4{margin:0 0 4px;font-size:14px}
.ovblock{margin:8px 0 14px}
.ovbar{display:flex;gap:8px;align-items:center;margin:0 0 6px;font-size:12px;color:#64748b}
.ovbar button{background:#fff;border:1px solid #b6c2d1;color:#1f3a5f;border-radius:6px;
              padding:5px 11px;font-size:13px;cursor:pointer}
.ovbar button:hover{background:#eef4ff}
.ovbox{position:relative;width:100%;background:#fff;border:1px solid #e2e8f0;border-radius:8px;overflow:hidden}
.ovbox img{position:absolute;left:50%;top:50%;transform-origin:center center;max-width:none;
           display:block;border:0}
td a.zoom{display:block}
.mono{font-family:Consolas,monospace;font-size:15px;letter-spacing:1px;white-space:nowrap}
.bad-ch{color:#c00;background:#ffe3e3;padding:0 2px;border-radius:3px}
.badge{color:#fff;padding:3px 9px;border-radius:11px;font-size:12px;white-space:nowrap;display:inline-block}
tr.r-NG{background:#fff8f8} tr.r-CHECK,tr.r-NOTEXT,tr.r-FORMAT{background:#fffdf5}
.why{color:#64748b;font-size:11.5px}
.hint{background:#fff7e6;border:1px solid #ffd591;border-radius:8px;padding:10px 14px;font-size:13px;margin:16px 0 0}
</style></head><body>
<header><h1>🏷️ 標籤條碼檢查</h1>
<p>把標籤稿 PDF 拖進來，開印前先驗一次：條碼解碼 vs 下方印的數字</p></header>
<div class="wrap">

  <div id="drop">
    <div class="big">把標籤 PDF 或照片拖到這裡</div>
    <div class="sm">也可以點一下選檔　·　可一次多個檔　·　支援 pdf / jpg / png / tif</div>
  </div>
  <input id="picker" type="file" multiple accept=".pdf,.png,.jpg,.jpeg,.tif,.tiff" hidden>

  <div class="opts">
    <label><b>應有條碼總數</b><input id="expect" type="number" min="1" placeholder="未填"></label>
    <span style="color:#94a3b8">不填就不做總數核對，也不會給「可以放行」</span>
    <span style="color:#94a3b8">或</span>
    <input id="path" type="text" placeholder="貼上電腦上的檔案路徑，報告會直接存在原檔旁邊">
    <button id="runpath" class="ghost">檢查這個路徑</button>
  </div>

  <div class="hint"><b>疊合對照怎麼看：</b>紅字＝<b>條碼解碼</b>出來的內容、藍字＝OCR 讀到的，兩者都逐字疊回原本印的字上
  （對位用字形切割，不經 OCR）。<b>紅字和黑字對不上＝條碼與印字真的不一致</b>，這條不依賴 OCR，可以直接用肉眼確認。<br>
  判定原則：<b>兩次 OCR 與完整模板辨識相符，才判定 OK</b>。
  缺讀、模板不完整或辨識結果衝突會標「需人工確認」；低信心有模板佐證時會註明依據。</div>

  <div id="status"></div>
  <div id="out"></div>
</div>
<script>
const $ = s => document.querySelector(s);
const drop = $('#drop'), picker = $('#picker'), out = $('#out'), status = $('#status');

drop.onclick = () => picker.click();
picker.onchange = e => handle([...e.target.files]);
;['dragenter','dragover'].forEach(t => drop.addEventListener(t, e => {
  e.preventDefault(); drop.classList.add('hot');
}));
;['dragleave','drop'].forEach(t => drop.addEventListener(t, e => {
  e.preventDefault(); drop.classList.remove('hot');
}));
drop.addEventListener('drop', e => handle([...e.dataTransfer.files]));
$('#runpath').onclick = async () => {
  const p = $('#path').value.trim();
  if (!p) return;
  busy('檢查中：' + p);
  try { render(await post('/api/check_path', JSON.stringify({path: p, expect: expectVal()}), 'application/json')); }
  catch (err) { fail(err); return; }
  done();
};

function expectVal(){
  const text = $('#expect').value.trim();
  if (!text) return null;
  const v = Number(text);
  if (!Number.isSafeInteger(v) || v < 1) throw new Error('應有條碼總數必須是大於 0 的整數');
  return v;
}
function busy(t){ status.innerHTML = '<span class="spin"></span>' + esc(t); }
function done(){ status.textContent = ''; }
function fail(e){ status.innerHTML = '<span style="color:#c00">出錯了：' + esc(String(e)) + '</span>'; }
function esc(s){ return (s||'').replace(/[&<>]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;'}[c])); }

function ovFit(img){
  const r = +(img.dataset.rot || 0), box = img.parentElement;
  const W = box.clientWidth - 16, nw = img.naturalWidth, nh = img.naturalHeight;
  if (!nw) return;
  const asp = nh / nw;
  const w = (r % 180 === 0) ? Math.min(nw, W) : Math.min(nw, W / asp);
  img.style.width = w + 'px';
  box.style.height = (((r % 180 === 0) ? w * asp : w) + 16) + 'px';
  img.style.transform = `translate(-50%,-50%) rotate(${r}deg)`;
}
window.addEventListener('resize', () => document.querySelectorAll('.ovbox img').forEach(ovFit));

async function post(url, body, type, extra){
  const r = await fetch(url, {method:'POST', headers: Object.assign({'Content-Type': type}, extra||{}), body});
  const j = await r.json();
  if (!j.ok) throw new Error(j.error || '未知錯誤');
  return j.data;
}

async function handle(files){
  files = files.filter(f => /\.(pdf|png|jpe?g|tiff?)$/i.test(f.name));
  if (!files.length) { fail('只吃 pdf / jpg / png / tif'); return; }
  for (let i = 0; i < files.length; i++){
    busy(`檢查中 (${i+1}/${files.length})：${files[i].name}　—　第一次會先載入 OCR 模型，請稍候`);
    try {
      const buf = await files[i].arrayBuffer();
      render(await post('/api/check', buf, 'application/octet-stream',
        {'X-Filename': encodeURIComponent(files[i].name), 'X-Expect': expectVal() ?? ''}));
    } catch (e) { fail(e); return; }
  }
  done();
}

function diff(a, b){
  const n = Math.max(a.length, b.length); let x = '', y = '';
  for (let i = 0; i < n; i++){
    const c1 = a[i] ?? '·', c2 = b[i] ?? '·';
    if (c1 === c2) { x += esc(c1); y += esc(c2); }
    else { x += '<b class="bad-ch">'+esc(c1)+'</b>'; y += '<b class="bad-ch">'+esc(c2)+'</b>'; }
  }
  return [x, y];
}
const BADGE = {OK:['OK','#0a7d32'], NG:['NG 不一致','#c00'], CHECK:['需人工確認','#b26a00'],
               NOTEXT:['讀不到文字','#b26a00'], FORMAT:['格式異常','#b26a00']};

function render(d){
  const el = document.createElement('div');
  el.className = 'file';
  const rows = d.rows.map(r => {
    const [a, b] = diff(r.bc, r.ocr);
    const [lb, col] = BADGE[r.verdict];
    const raw = (r.raw||'').trim();
    return `<tr class="r-${r.verdict}">
      <td>${r.page}-${r.label > 0 ? r.label : '-'}</td>
      <td><img src="${r.img}"></td>
      <td>
        <div><span class="tag t-bc">條碼</span><span class="mono">${a}</span></div>
        <div><span class="tag t-ocr">OCR</span><span class="mono">${b}</span></div>
        <div><span class="tag t-sh">字形</span><span class="mono">${esc(r.shape||'—')}</span>
          <span class="why">　${r.shape ? Math.round(r.shape_conf*100)+'%' : ''}</span></div>
        ${raw && raw !== r.ocr ? '<div class="why">原文：'+esc(raw)+'</div>' : ''}</td>
      <td>${r.cmp ? `<a class="zoom" href="${r.cmp}" target="_blank" title="點開看原尺寸"><img class="cmp" src="${r.cmp}"></a>` : '—'}</td>
      <td><span class="badge" style="background:${col}">${lb}</span></td>
      <td class="why">${esc(r.reason)}
        ${r.tpl_note ? (() => {
          const bad = (r.tpl_bad && r.tpl_bad.length) || r.tpl_note.includes('分不出來');
          const cls = bad ? 'tplbad' : (r.tpl_weak ? 'tplweak' : 'tplok');
          const mk = bad ? '⚠ ' : (r.tpl_weak ? '△ ' : '✓ ');
          return `<div class="${cls}">${mk}${esc(r.tpl_note)}</div>`; })() : ''}
        <br>${esc(r.src)}　信心 ${r.score.toFixed(2)}　${esc(r.fmt)} / ${r.orient}°</td>
    </tr>`;
  }).join('');
  el.innerHTML = `
    <div class="fhead"><span class="fname">${esc(d.name)}</span>
      ${d.ng_lines.length ? '<button class="ghost" data-copy>複製異常清單</button>' : ''}
      <button class="ghost" data-save>下載報告 HTML</button>
      <button class="ghost" data-pdf>下載 PDF</button></div>
    <div class="banner ${d.tone}">${esc(d.headline)}</div>
    <div class="stats">
      <div class="stat">條碼總數<b>${d.total}</b></div>
      <div class="stat s-ok">OK<b>${d.ok}</b></div>
      <div class="stat s-ng">NG 不一致<b>${d.ng}</b></div>
      <div class="stat s-ck">需人工確認<b>${d.check}</b></div>
      <div class="stat">料號<b style="font-size:13px;line-height:1.5">${d.codes.map(esc).join('<br>')}</b></div>
    </div>
    ${(d.rows.filter(r => r.overview).map(r =>
        `<div class="ov"><img src="${r.overview}"></div>`).join('')) ?
      `<div class="ovwrap"><h4>原圖對照頁</h4>
       <p class="why">整頁原樣，紅字是<b>條碼解碼出來的內容</b>，貼在該標籤印刷數字的正下方，方向跟著標籤本身走。
       <b>綠框＝相符、紅框＋叉＝不符、橘框＋問號＝需人工確認</b>（三者意思不同，請看下方表格的原因）。</p>
       ${d.rows.filter(r => r.overview).map(r => `
         <div class="ovblock">
           <div class="ovbar">
             <button type="button" data-rot="-90">↺ 左轉 90°</button>
             <button type="button" data-rot="90">↻ 右轉 90°</button>
             <button type="button" data-rot="0">回正</button>
             <span>第 ${r.page} 頁　·　旋轉不影響判定，只是方便你照實際貼標方向核對</span>
           </div>
           <div class="ovbox"><img src="${r.overview}"></div>
         </div>`).join('')}</div>` : ''}
    ${d.ng_lines.length ? `<div class="ngbox"><h4>異常清單（可直接複製回報）</h4><pre>${esc(d.ng_lines.join('\n'))}</pre></div>` : ''}
    ${d.warns.length ? `<ul class="warns">${d.warns.map(w => '<li>'+esc(w)+'</li>').join('')}</ul>` : ''}
    <div class="tw"><table><tr><th>頁-標籤</th><th>條碼與文字</th>
      <th>三種判讀值<br><span class="why">條碼=解碼　OCR=辨識　字形=多實例覆核</span></th>
      <th>疊合對照<br><span class="why">紅=條碼字　藍=OCR字，疊回原印字上</span></th>
      <th>判定</th><th>依據</th></tr>${rows}</table></div>`;
  el.querySelectorAll('.ovbar button').forEach(b => b.onclick = () => {
    const img = b.closest('.ovblock').querySelector('img');
    const d = +b.dataset.rot;
    img.dataset.rot = d ? ((+(img.dataset.rot || 0) + d) % 360 + 360) % 360 : 0;
    ovFit(img);
  });
  el.querySelectorAll('.ovbox img').forEach(im => {
    if (im.complete) ovFit(im); else im.onload = () => ovFit(im);
  });
  const c = el.querySelector('[data-copy]');
  if (c) c.onclick = () => { navigator.clipboard.writeText(d.ng_lines.join('\n')); c.textContent = '已複製 ✓'; };
  const pb = el.querySelector('[data-pdf]');
  pb.onclick = async () => {
    const t = pb.textContent;
    status.textContent = '';
    pb.textContent = '轉檔中…'; pb.disabled = true;
    try {
      const r = await post('/api/pdf', JSON.stringify({name: d.name, html: d.report_html}),
                           'application/json');
      const bin = Uint8Array.from(atob(r.pdf), c => c.charCodeAt(0));
      const a = document.createElement('a');
      a.href = URL.createObjectURL(new Blob([bin], {type: 'application/pdf'}));
      a.download = r.name;
      a.click();
      pb.textContent = '已下載 ✓';
    } catch (e) { pb.textContent = 'PDF 失敗'; fail(e); }
    finally { pb.disabled = false; setTimeout(() => pb.textContent = t, 2500); }
  };
  el.querySelector('[data-save]').onclick = () => {
    const blob = new Blob([d.report_html], {type: 'text/html;charset=utf-8'});
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = d.name.replace(/\.[^.]+$/, '') + '_條碼檢查報告.html';
    a.click();
  };
  out.prepend(el);
}
fetch('/api/warmup');
</script></body></html>"""


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *a):
        pass

    def _send(self, code, body, ctype="application/json; charset=utf-8"):
        if isinstance(body, str):
            body = body.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")   # 程式更新後不要拿到舊頁面
        self.end_headers()
        self.wfile.write(body)

    def _json(self, data=None, error=None):
        self._send(200 if error is None else 200,
                   json.dumps({"ok": error is None, "data": data, "error": error}, ensure_ascii=False))

    def do_GET(self):
        p = urllib.parse.urlparse(self.path).path
        if p == "/":
            self._send(200, PAGE, "text/html; charset=utf-8")
        elif p == "/api/warmup":
            threading.Thread(target=warmup, daemon=True).start()
            self._json({"warming": True})
        else:
            self._send(404, "not found", "text/plain; charset=utf-8")

    def do_POST(self):
        p = urllib.parse.urlparse(self.path).path
        n = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(n) if n else b""
        try:
            if p == "/api/check":
                name = urllib.parse.unquote(self.headers.get("X-Filename", "label.pdf"))
                name = os.path.basename(name).replace("\\", "_")
                exp = self.headers.get("X-Expect", "").strip()
                os.makedirs(UPLOAD_DIR, exist_ok=True)
                path = os.path.join(UPLOAD_DIR, "%s_%s" % (uuid.uuid4().hex, name))
                with open(path, "wb") as f:
                    f.write(body)
                data = run_check(path, int(exp) if exp else None)
                data["name"] = name
                self._json(data)
            elif p == "/api/check_path":
                req = json.loads(body.decode("utf-8"))
                path = req.get("path", "").strip().strip('"')
                if not os.path.isfile(path):
                    self._json(error="找不到這個檔案：" + path)
                    return
                data = run_check(path, req.get("expect"))
                out = os.path.splitext(path)[0] + "_條碼檢查報告.html"
                try:
                    with open(out, "w", encoding="utf-8") as f:
                        f.write(data["report_html"])
                    data["saved"] = out
                    data["warns"] = list(data["warns"]) + ["報告已存到：" + out]
                except Exception as e:
                    data["warns"] = list(data["warns"]) + ["報告存檔失敗：%s" % e]
                self._json(data)
            elif p == "/api/pdf":
                req = json.loads(body.decode("utf-8"))
                stem = os.path.splitext(os.path.basename(req.get("name", "報告")))[0]
                work = tempfile.mkdtemp(prefix="labelpdfweb_")
                try:
                    hp = os.path.join(work, "r.html")
                    with open(hp, "w", encoding="utf-8") as f:
                        f.write(req.get("html", ""))
                    pp = os.path.join(work, "r.pdf")
                    C.html_to_pdf(hp, pp)
                    with open(pp, "rb") as f:
                        blob = base64.b64encode(f.read()).decode()
                    self._json({"pdf": blob, "name": stem + "_條碼檢查報告.pdf"})
                finally:
                    shutil.rmtree(work, ignore_errors=True)
            else:
                self._json(error="unknown endpoint")
        except Exception as e:
            traceback.print_exc()
            self._json(error="%s: %s" % (type(e).__name__, e))


def prune_uploads(keep_days=7):
    """拖進來的檔會暫存在 _uploads，開機時把舊的清掉，別長期佔空間。"""
    if not os.path.isdir(UPLOAD_DIR):
        return
    cut = time.time() - keep_days * 86400
    for n in os.listdir(UPLOAD_DIR):
        f = os.path.join(UPLOAD_DIR, n)
        try:
            if os.path.isfile(f) and os.path.getmtime(f) < cut:
                os.remove(f)
        except OSError:
            pass


def main():
    prune_uploads()
    url = "http://localhost:%d" % PORT
    try:
        srv = ThreadingHTTPServer(("127.0.0.1", PORT), Handler)
    except OSError:
        # 已經有一個在跑了。以前這裡會默默失敗，然後你在瀏覽器看到的是舊版程式。
        print("port %d 已經被占用——應該是先前開的那一個還在跑。" % PORT)
        print("如果程式有更新，請先關掉舊的那個黑色視窗（或重開機）再啟動。")
        print("要直接用舊的，開這個網址就好：" + url)
        return 2
    print("標籤條碼檢查已啟動：" + url)
    print("（關掉這個視窗就會停止服務）")
    threading.Thread(target=warmup, daemon=True).start()
    if "--no-browser" not in sys.argv:
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
