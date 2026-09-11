# -*- coding: utf-8 -*-
"""
標籤條碼檢查工具
把標籤稿 PDF（或實體標籤照片）丟進來，逐一解出每個條碼的實際內容，
再跟條碼下方印的人眼可讀文字比對，找出不一致。

用法：
    python check_labels.py 檔案.pdf [檔案2.pdf ...] [--expect 應有條碼總數] [--pdf]
    不帶參數會跳出選檔視窗。

輸出：與來源檔同資料夾的 "<檔名>_條碼檢查報告.html"，並自動開啟。
全程離線執行，不連網。

判定原則：不確定就說不確定。OK 只給「有兩條互相獨立的證據支持」的列，
其餘一律 CHECK，並在報告寫清楚是哪一種不確定。
"""
import sys, os, io, re, math, base64, datetime, webbrowser, unicodedata
from html import escape
import threading

try:
    import pymupdf as fitz            # PyMuPDF
    import numpy as np
    from PIL import Image
    import zxingcpp
    try:
        import cv2                   # 字形覆核用；沒有也能跑，但該列會標 CHECK
    except ImportError:
        cv2 = None
except ImportError as e:
    print("缺少套件：", e)
    print('請執行： python -m pip install pymupdf pillow numpy zxing-cpp rapidocr_onnxruntime opencv-python')
    sys.exit(1)

# ── 可調設定 ──────────────────────────────────────────────
DPI = 400              # 主要解析度
DPI_HI = 800           # 第二次獨立判讀的解析度
MIN_SCORE = 0.80       # OCR 信心門檻，低於此值一律 CHECK
GLYPH_PURITY = 0.80    # 字形群聚多數決的最低得票比例
VOTE_MIN = 2           # 一個字形至少要被幾個實例投票才採信
MAX_SKEW = 5.0         # 條碼傾斜超過幾度就不判讀（拍歪的照片）
CODE_RE = re.compile(r"^\d{9}A\d{8}$")   # 料號格式：9碼數字 + A + 8碼日期

# 比對前的正規化——預設只忽略空白（因為空白是我們自己組字串時加的），
# 大小寫與標點「不」忽略，要忽略請自行打開，報告一律顯示原文。
IGNORE_SPACE = True
IGNORE_CASE = False
IGNORE_PUNCT = False
# ─────────────────────────────────────────────────────────

_OCR = {"e": None}
_OCR_LOCK = threading.RLock()


def ocr_engine():
    with _OCR_LOCK:
        if _OCR["e"] is None:
            from rapidocr_onnxruntime import RapidOCR
            print("  載入 OCR 模型…")
            _OCR["e"] = RapidOCR()
        return _OCR["e"]


def fold_fullwidth(s):
    """全形折成半形（NFKC）。OCR 模型的字典同時收全形與半形，影像品質差時
    會吐出全形數字（２7７０…），不折的話會被當成「跟條碼不一致」而誤報。
    逐字折、且只接受「一個字換一個字而且換出來是 ASCII」的結果，避免
    ㎏→kg 這類相容分解改變字數，破壞後面字形票數的逐字對位。"""
    out = []
    for ch in s:
        c = unicodedata.normalize("NFKC", ch)
        out.append(c if len(c) == 1 and c.isascii() else ch)
    return "".join(out)


def norm(s):
    """比對用的正規化。先把全形折成半形，預設只拿掉空白，不動大小寫與標點。"""
    s = fold_fullwidth(s or "")
    if IGNORE_SPACE:
        s = re.sub(r"\s+", "", s)
    if IGNORE_CASE:
        s = s.upper()
    if IGNORE_PUNCT:
        s = re.sub(r"[^0-9A-Za-z]", "", s)
    return s


def render(page, dpi):
    pix = page.get_pixmap(dpi=dpi)
    img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples).convert("L")
    scale = pix.width / page.rect.width if page.rect.width else dpi / 72.0
    return img, scale


# ── 幾何：一律從條碼自己的四個角推算，不用寫死的方向對照表 ──

def quad_pts(res):
    p = res.position
    return np.array([[p.top_left.x, p.top_left.y], [p.top_right.x, p.top_right.y],
                     [p.bottom_right.x, p.bottom_right.y], [p.bottom_left.x, p.bottom_left.y]], float)


def text_geom(q, W, H):
    """用條碼四角算出「它自己下方」那塊文字區，以及要轉正需旋轉幾度。
    回傳 (矩形, 旋轉角, 傾斜量)。

    注意：zxing 的 top/bottom 是**頁面座標**的上下，只有 top_left→top_right
    這條邊代表條碼真正的閱讀方向。所以垂直方向必須由閱讀方向轉 90° 求得，
    不能直接拿 bottom_left−top_left（180° 的條碼會算到反邊）。"""
    right = q[1] - q[0]
    rn = np.linalg.norm(right)
    if rn < 1:
        return None, 0.0, 99.0
    r_hat = right / rn
    d_hat = np.array([-r_hat[1], r_hat[0]])          # 條碼自己的「下方」
    rp, dp = q @ r_hat, q @ d_hat
    width, thick = rp.max() - rp.min(), dp.max() - dp.min()
    if thick < 1:
        return None, 0.0, 99.0
    gap, ext, side = thick * 0.05, thick * 1.15, width * 0.06
    corners = np.array([r_hat * a + d_hat * b
                        for a in (rp.min() - side, rp.max() + side)
                        for b in (dp.max() + gap, dp.max() + gap + ext)])
    x0, y0 = corners.min(0)
    x1, y1 = corners.max(0)
    rect = (max(0.0, x0), max(0.0, y0), min(float(W), x1), min(float(H), y1))
    ang = math.degrees(math.atan2(r_hat[1], r_hat[0]))
    snapped = round(ang / 90.0) * 90.0
    return rect, snapped, abs(ang - snapped)


def upright(img, ang):
    a = int(round(ang)) % 360
    return img if a == 0 else img.rotate(a, expand=True)


def read_pdf_words(page, rect_px, scale, ang):
    """取畫面區域內的文字層，作為輔助；文字層可能隱藏或與畫面不同。"""
    r = fitz.Rect(rect_px[0] / scale, rect_px[1] / scale, rect_px[2] / scale, rect_px[3] / scale)
    words = []
    for w in page.get_text("words"):
        visible_rect = fitz.Rect(w[:4]) * page.rotation_matrix
        if visible_rect.intersects(r):
            words.append((*tuple(visible_rect), *w[4:]))
    if not words:
        return None
    a = int(round(ang)) % 360
    if a in (0, 180):
        words.sort(key=lambda w: (round(w[1], 1), w[0]))
    else:
        words.sort(key=lambda w: (round(w[0], 1), w[1]))
    if a in (180, 270):
        words = words[::-1]
    return " ".join(w[4] for w in words)


def ocr_text(img):
    with _OCR_LOCK:
        res, _ = ocr_engine()(np.array(img.convert("RGB")))
    if not res:
        return "", 0.0
    res = sorted(res, key=lambda r: r[0][0][0])
    txt = "".join(r[1] for r in res)
    score = sum(float(r[2]) for r in res) / len(res)
    return txt, score


def to_b64(img, maxw=900):
    if img.width > maxw:
        img = img.resize((maxw, max(1, int(img.height * maxw / img.width))))
    buf = io.BytesIO()
    img.save(buf, "PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def label_boxes(page, scale):
    """抓 PDF 裡「畫出來的」標籤外框（只取線框，不取填色區塊），用來把條碼分組成同一張標籤。"""
    out = []
    m = page.rotation_matrix          # get_drawings 給的是未旋轉座標，要轉到畫面座標
    for it in page.get_drawings():
        if it.get("type") != "s":            # s = stroke（線框）
            continue
        r = it["rect"] * m
        if r.width * scale > 120 and r.height * scale > 60:
            out.append((min(r.x0, r.x1) * scale, min(r.y0, r.y1) * scale,
                        max(r.x0, r.x1) * scale, max(r.y0, r.y1) * scale))
    out.sort(key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))
    return out


def owner_label(box, boxes):
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    for i, b in enumerate(boxes):
        if b[0] <= cx <= b[2] and b[1] <= cy <= b[3]:
            return i
    return -1


def diff_html(bc, tx):
    n = max(len(bc), len(tx))
    a = b = ""
    for i in range(n):
        c1 = bc[i] if i < len(bc) else "·"
        c2 = tx[i] if i < len(tx) else "·"
        if c1 == c2:
            a += escape(c1)
            b += escape(c2)
        else:
            a += '<b class="bad">' + escape(c1) + '</b>'
            b += '<b class="bad">' + escape(c2) + '</b>'
    return a, b


# ── 字形覆核（輔助手段：票源仍是 OCR，靠多實例互相檢查，不是獨立辨識器） ──

GLYPH_N = 24
# 門檻是量出來的，不是猜的：把整份稿的字依 OCR 標記分類後量相似度，
# 全檔混在一起時「同字最低 0.876 / 異字最高 0.915」是重疊的，單一門檻分不開；
# 但限定同一字級內，同字最低 0.926~0.946、異字最高 0.874~0.915，中間有空隙。
# 所以分群一律先按字級分組，門檻取空隙中間值。字級不同的同一個字各自成群沒關係，
# 分開只是少共用票，不會判錯；merge 錯才會出事。
SIM_TH = 0.92
SIZE_LO, SIZE_HI = 0.80, 1.25      # 同一群允許的字高比例範圍


def segment_glyphs(img):
    if cv2 is None:
        return []
    a = np.array(img)
    if a.size == 0:
        return []
    _t, bw = cv2.threshold(a, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    n, _lab, stats, _c = cv2.connectedComponentsWithStats(bw, 8)
    IH, IW = a.shape
    comps = []
    for i in range(1, n):
        c = stats[i]
        x, y, w, h, area = (c[cv2.CC_STAT_LEFT], c[cv2.CC_STAT_TOP], c[cv2.CC_STAT_WIDTH],
                            c[cv2.CC_STAT_HEIGHT], c[cv2.CC_STAT_AREA])
        if h > IH * 0.85 or w > IW * 0.4 or w > h * 3 or h < 3 or area < 12:
            continue                      # 標籤外框直線、長橫線、雜點
        comps.append(c)
    if not comps:
        return []
    # 字高基準取「同高元件最多的那一群」，不用中位數。
    # 中位數會被裁進來的外框線與條碼底部餘料帶歪（實測整排數字因此被濾掉），
    # 但一整排數字永遠是同高元件最多的一群。
    hs = [c[cv2.CC_STAT_HEIGHT] for c in comps]
    hmed, best = hs[0], 0
    for h0 in hs:
        cnt = sum(1 for h in hs if h0 * 0.8 <= h <= h0 * 1.25)
        if cnt > best:
            hmed, best = h0, cnt
    out = []
    for c in comps:
        x, y, w, h, area = (c[cv2.CC_STAT_LEFT], c[cv2.CC_STAT_TOP], c[cv2.CC_STAT_WIDTH],
                            c[cv2.CC_STAT_HEIGHT], c[cv2.CC_STAT_AREA])
        if h < hmed * 0.55 or h > hmed * 1.8 or w > hmed * 2.5:
            continue
        g = bw[y:y + h, x:x + w]
        side = max(w, h)
        pad = np.zeros((side, side), np.uint8)
        pad[(side - h) // 2:(side - h) // 2 + h, (side - w) // 2:(side - w) // 2 + w] = g
        v = cv2.resize(pad, (GLYPH_N, GLYPH_N), interpolation=cv2.INTER_AREA).astype(np.float32).ravel()
        nv = np.linalg.norm(v)
        if nv < 1e-6:
            continue
        out.append((int(x), v / nv, int(h), int(y), int(w)))
    out.sort(key=lambda t: t[0])
    return out


def shape_read(rows):
    """把整份檔的字依形狀分群，同一形狀的所有實例用 OCR 票多數決定它是哪個字。
    票數不足、平手、得票比例不足 → 該字標「?」，該列後續會判 CHECK。"""
    cents, counts, members, sizes = [], [], [], []
    for ri, r in enumerate(rows):
        for gi, g in enumerate(r.get("glyphs", [])):
            v, h = g[1], g[2]
            best, bi = -1.0, -1
            for ci, c in enumerate(cents):
                ratio = h / sizes[ci] if sizes[ci] else 0
                if not (SIZE_LO <= ratio <= SIZE_HI):      # 字級不同就不比，避免跨字級誤併
                    continue
                s = float(np.dot(v, c))
                if s > best:
                    best, bi = s, ci
            if best >= SIM_TH:
                k = counts[bi]
                cents[bi] = (cents[bi] * k + v) / (k + 1)
                cents[bi] /= max(np.linalg.norm(cents[bi]), 1e-6)
                sizes[bi] = (sizes[bi] * k + h) / (k + 1)
                counts[bi] += 1
                members[bi].append((ri, gi))
            else:
                cents.append(v.copy())
                counts.append(1)
                sizes.append(float(h))
                members.append([(ri, gi)])

    votes = [{} for _ in cents]
    for ci, mem in enumerate(members):
        for ri, gi in mem:
            t = rows[ri].get("ocr", "")
            if len(rows[ri].get("glyphs", [])) == len(t) and gi < len(t):
                votes[ci][t[gi]] = votes[ci].get(t[gi], 0) + 1

    labels, confs = [], []
    for v in votes:
        if not v:
            labels.append("?")
            confs.append(0.0)
            continue
        ordered = sorted(v.items(), key=lambda kv: -kv[1])
        ch, cnt = ordered[0]
        tot = sum(v.values())
        tie = len(ordered) > 1 and ordered[1][1] == cnt
        purity = cnt / tot
        good = (cnt >= VOTE_MIN) and (not tie) and (purity >= GLYPH_PURITY)
        labels.append(ch if good else "?")
        confs.append(purity)

    of = {}
    for ci, mem in enumerate(members):
        for ri, gi in mem:
            of[(ri, gi)] = ci
    for ri, r in enumerate(rows):
        gl = r.get("glyphs", [])
        if not gl:
            r["shape"], r["shape_conf"] = "", 0.0
            continue
        chars, cf = [], []
        for gi in range(len(gl)):
            ci = of.get((ri, gi))
            chars.append(labels[ci] if ci is not None else "?")
            cf.append(confs[ci] if ci is not None else 0.0)
        r["shape"] = "".join(chars)
        r["shape_conf"] = min(cf) if cf else 0.0
        r["glyph_labels"] = list(zip(chars, cf))
    return len(cents)


# ── 疊合對照圖：把「條碼解出的字」與「OCR 讀到的字」逐字對位畫回印刷字上 ──
# 對位靠字形切割（不經 OCR），條碼那條的字來自解碼（也不經 OCR），
# 所以最上面那條帶子可以完全不依賴 OCR，用肉眼直接確認一致或不一致。

FONT_PATHS = [r"C:\Windows\Fonts\arial.ttf", r"C:\Windows\Fonts\arialbd.ttf"]
LABEL_FONT = r"C:\Windows\Fonts\msjh.ttc"
CLR_BC = (214, 0, 0)          # 條碼解出的字：紅
CLR_OCR = (0, 82, 204)        # OCR 讀到的字：藍
CLR_DIFF = (214, 0, 0)        # 兩者不同的位置：紅框
CLR_DOUBT = (230, 145, 0)     # 程式自己也不確定的字：橘框
_FONTS = {}


def _font(size, label=False):
    key = (size, label)
    if key not in _FONTS:
        from PIL import ImageFont
        path = LABEL_FONT if label else next((p for p in FONT_PATHS if os.path.exists(p)), None)
        try:
            _FONTS[key] = ImageFont.truetype(path, size)
        except Exception:
            _FONTS[key] = ImageFont.load_default()
    return _FONTS[key]


def _draw_char(d, ch, box, color, alpha):
    """把一個字畫進指定的字框裡：字高對齊、水平置中。"""
    x, y, w, h = box
    f = _font(max(8, int(h / 0.72)))
    bb = d.textbbox((0, 0), ch, font=f)
    cw, chh = bb[2] - bb[0], bb[3] - bb[1]
    d.text((x + (w - cw) / 2 - bb[0], y + (h - chh) / 2 - bb[1]), ch, font=f,
           fill=color + (alpha,))


# ── 模板疊合比對：把條碼說的那個字渲染出來，直接跟印刷字比形狀 ──
# 這條完全不經 OCR：它不問「這是什麼字」，只問「這個形狀像不像條碼說的那個字」。
# 實測 144 個字 top-1 全中，最佳與次佳的相似度差距最小 0.056，所以邊界很清楚。
TPL_CAND = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
TPL_MIN_MARGIN = 0.02          # 最佳與次佳太接近就當作分不出來
_TPL = {}


def _template(ch, h):
    key = (ch, int(round(h / 4.0)) * 4)
    if key in _TPL:
        return _TPL[key]
    from PIL import ImageDraw
    hh = key[1]
    f = _font(max(8, int(hh / 0.72)))
    im = Image.new("L", (int(hh * 2.2) + 8, int(hh * 2.2) + 8), 0)
    d = ImageDraw.Draw(im)
    bb = d.textbbox((0, 0), ch, font=f)
    d.text((-bb[0] + int(hh * 0.4), -bb[1] + int(hh * 0.4)), ch, font=f, fill=255)
    a = np.array(im)
    ys, xs = np.where(a > 96)
    v = None
    if len(xs):
        g = a[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
        gh, gw = g.shape
        side = max(gh, gw)
        pad = np.zeros((side, side), np.uint8)
        pad[(side - gh) // 2:(side - gh) // 2 + gh, (side - gw) // 2:(side - gw) // 2 + gw] = g
        v = cv2.resize(pad, (GLYPH_N, GLYPH_N), interpolation=cv2.INTER_AREA).astype(np.float32).ravel()
        v = np.where(v >= 128, 255.0, 0.0).astype(np.float32)
        n = np.linalg.norm(v)
        v = v / n if n > 1e-6 else None
    _TPL[key] = v
    return v


def template_read(glyphs):
    """不靠 OCR，直接用形狀比對讀出每個字；回傳 (讀出的字串, 每個字的把握度)。"""
    if cv2 is None or not glyphs:
        return "", []
    out, margins = [], []
    for g in glyphs:
        vec, h = g[1], g[2]
        sims = []
        for c in TPL_CAND:
            t = _template(c, h)
            if t is not None:
                sims.append((float(np.dot(vec, t)), c))
        if len(sims) < 2:
            out.append("?")
            margins.append(0.0)
            continue
        sims.sort(reverse=True)
        margin = sims[0][0] - sims[1][0]
        out.append(sims[0][1] if margin >= TPL_MIN_MARGIN else "?")
        margins.append(margin)
    return "".join(out), margins


def compare_image(crop, glyphs, bc, ocr, glyph_labels, tpl="", tpl_bad=None):
    """三條帶：①印字＋條碼字疊合 ②印字＋OCR字疊合 ③條碼字／OCR字對位排列。"""
    from PIL import ImageDraw
    if not glyphs:
        return None
    boxes = [(g[0], g[3], g[4], g[2]) for g in glyphs]
    hmed = sorted(b[3] for b in boxes)[len(boxes) // 2]
    # 只留文字那一塊，上下左右多餘的留白（含裁進來的標籤外框線）都切掉
    m = int(hmed * 0.30)
    top = max(0, min(b[1] for b in boxes) - m)
    bot = min(crop.height, max(b[1] + b[3] for b in boxes) + m)
    lft = max(0, min(b[0] for b in boxes) - m)
    rgt = min(crop.width, max(b[0] + b[2] for b in boxes) + m)
    base = crop.convert("RGB").crop((lft, top, rgt, bot))
    boxes = [(x - lft, y - top, w, h) for (x, y, w, h) in boxes]
    W, H = base.size
    # 左側標籤欄：字級跟著印字大小走，欄寬量出來（報告裡圖會縮到約 0.57 倍，太小會看不清楚）
    lsize = max(18, int(hmed * 0.68))
    lfont = _font(lsize, label=True)
    _md = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    pad = int(max(_md.textlength(t, font=lfont) for t in ("條碼疊合", "OCR疊合", "三排對位")) + 16)
    gapy = 10                                  # 帶與帶之間的空隙

    def band(text, color):
        layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        d = ImageDraw.Draw(layer)
        for i, b in enumerate(boxes):
            if i < len(text):
                _draw_char(d, text[i], b, color, 165)
        return Image.alpha_composite(base.convert("RGBA"), layer).convert("RGB")

    rowh = int(hmed * 1.30)
    strip = Image.new("RGB", (W, rowh * 3), "white")
    strip.paste(base, (0, int((rowh - H) / 2)))          # 第一排＝原本印的字
    ds = ImageDraw.Draw(strip)
    for i, b in enumerate(boxes):
        if i < len(bc):
            _draw_char(ds, bc[i], (b[0], rowh + (rowh - hmed) // 2, b[2], hmed), CLR_BC, 255)
        if i < len(ocr):
            _draw_char(ds, ocr[i], (b[0], rowh * 2 + (rowh - hmed) // 2, b[2], hmed), CLR_OCR, 255)

    bands = [("條碼疊合", band(bc, CLR_BC), H), ("OCR疊合", band(ocr, CLR_OCR), H),
             ("三排對位", strip, rowh * 3)]
    out = Image.new("RGB", (W + pad, sum(h for _t, _i, h in bands) + gapy * 2), "white")
    d = ImageDraw.Draw(out)
    tops, y = [], 0
    for i, (txt, im, h) in enumerate(bands):
        out.paste(im, (pad, y))
        d.text((6, y + h / 2 - lsize * 0.62), txt, font=lfont,
               fill=(CLR_BC, CLR_OCR, (90, 90, 90))[i])
        tops.append(y)
        y += h + (gapy if i < 2 else 0)

    # 標出「條碼與 OCR 判讀不同」與「程式自己也不確定」的字
    tb = set(tpl_bad or [])
    for i, b in enumerate(boxes):
        # 紅框＝模板比對（不經 OCR）判定這個字與條碼不符；沒有模板結果時退回字串比對
        bad = (i in tb) if tpl else (i < len(bc) and i < len(ocr) and bc[i] != ocr[i])
        ch, cf = glyph_labels[i] if i < len(glyph_labels) else ("?", 0.0)
        doubt = (ch == "?") or (cf < 0.95) or (i < len(ocr) and ch != ocr[i])             or (i < len(tpl) and tpl[i] == "?")
        if not (bad or doubt):
            continue
        col, wdt = (CLR_DIFF, 4) if bad else (CLR_DOUBT, 3)
        for t, hh in ((tops[0], H), (tops[1], H)):
            d.rectangle([pad + b[0] - 3, t + b[1] - 3,
                         pad + b[0] + b[2] + 3, t + b[1] + b[3] + 3], outline=col, width=wdt)
        d.rectangle([pad + b[0] - 3, tops[2], pad + b[0] + b[2] + 3, tops[2] + rowh * 3 - 1],
                    outline=col, width=wdt)
    return out


# ── 原圖對照頁：整頁原樣，在每個印刷數字正下方補上條碼解出來的數字 ──
# 方向跟著標籤本身走（原稿顛倒的，補上去的數字也跟著顛倒），
# 讓人可以在真實版面上、用原本的閱讀方向再核一次。

def overview_image(page_img, geoms, maxw=1700):
    """整頁原圖，在每個印刷數字正下方補上條碼解出來的數字。
    位置用實際切出來的字來定位（不是用文字區帶），否則會掉到下一張標籤上；
    方向跟著標籤本身走，原稿顛倒的補字也跟著顛倒。"""
    from PIL import ImageDraw
    base = page_img.convert("RGB")
    W, H = base.size
    layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    for treg, ang, ext, gboxes, text, tag, status in geoms:
        bad = status == 'ng'
        a = math.radians(ang)
        r_hat = np.array([math.cos(a), math.sin(a)])
        d_hat = np.array([-r_hat[1], r_hat[0]])
        corners = np.array([[treg[0], treg[1]], [treg[2], treg[1]],
                            [treg[2], treg[3]], [treg[0], treg[3]]])
        us, vs = corners @ r_hat, corners @ d_hat
        origin = corners[int(np.argmin(us + vs))]        # 條碼自身座標的原點（左上）
        gboxes = gboxes or []
        if ext:
            gx0, gx1, gy0, gy1 = ext
        else:                                            # 沒切到字（PDF 文字層那條路）就估一個
            LL, BB = us.max() - us.min(), vs.max() - vs.min()
            gx0, gx1, gy0, gy1 = LL * 0.08, LL * 0.92, BB * 0.12, BB * 0.45
        th, tw = max(6.0, gy1 - gy0), max(10.0, gx1 - gx0)
        # 逐字對齊：每個紅字畫在對應印刷字的正下方，字級以「字距」為上限，
        # 這樣整排數字會與上方印刷字一欄一欄對齊，比整串置中好比對得多。
        probe = ImageDraw.Draw(Image.new("RGB", (1, 1)))
        n = min(len(text), len(gboxes)) if gboxes else 0
        if n:
            pitch = min((gboxes[i + 1][0] - gboxes[i][0] for i in range(n - 1)),
                        default=tw / max(1, n))
            fs = max(9, int(th * 1.45))
            while fs > 9 and max(probe.textlength(c, font=_font(fs)) for c in set(text[:n])) > pitch * 0.94:
                fs -= 1
        else:
            fs = max(9, int(th * 1.2))
        f = _font(fs)
        pad = max(3, int(th * 0.34))
        tagf = _font(max(9, int(th * 0.85)), label=True)
        tb = probe.textbbox((0, 0), tag, font=tagf)
        gapx = max(3, int(th * 0.28))
        xmark = th * 0.80 if status != "ok" else 0.0
        L = int((tb[2] - tb[0]) + gapx * 2 + xmark)      # 框外左側：編號＋叉
        B = int(tw + pad * 2)                            # 框本身：剛好罩住整排數字
        sh = int(max(th * 2.0, fs * 1.5))
        strip = Image.new("RGBA", (L + B, sh), (0, 0, 0, 0))
        ds = ImageDraw.Draw(strip)
        # 綠＝已確認相符、紅＝確定不符、橘＝還沒確認（三者意思不同，別共用同一個顏色）
        boxc = {"ng": CLR_DIFF, "check": CLR_DOUBT}.get(status, (10, 125, 50))
        ds.rectangle([L, 0, L + B - 1, sh - 1], fill=(255, 255, 255, 236),
                     outline=boxc, width=max(2, fs // (9 if status != "ok" else 16)))
        cy0 = sh / 2.0
        ds.text((gapx, cy0 - (tb[3] - tb[1]) / 2 - tb[1]), tag, font=tagf, fill=(110, 110, 110, 255))
        if bad:                                          # 叉也放框外，框內全部留給數字
            m = th * 0.30
            cx0 = gapx + (tb[2] - tb[0]) + gapx + m
            for dy in (-1, 1):
                ds.line([cx0 - m, cy0 + m * dy, cx0 + m, cy0 - m * dy],
                        fill=CLR_DIFF + (255,), width=max(2, int(th * 0.13)))
        elif status == "check":                          # 待確認：橘色問號，別跟「不符」混淆
            qf = _font(max(10, int(th * 1.2)))
            qb = ds.textbbox((0, 0), "?", font=qf)
            ds.text((gapx + (tb[2] - tb[0]) + gapx, cy0 - (qb[3] - qb[1]) / 2 - qb[1]),
                    "?", font=qf, fill=CLR_DOUBT + (255,))
        if n:
            for i in range(n):
                gx, gw = gboxes[i][0], gboxes[i][1]
                cb = ds.textbbox((0, 0), text[i], font=f)
                ds.text((L + pad + (gx - gx0) + (gw - (cb[2] - cb[0])) / 2 - cb[0],
                         cy0 - (cb[3] - cb[1]) / 2 - cb[1]), text[i], font=f, fill=CLR_BC + (255,))
        else:
            cb = ds.textbbox((0, 0), text, font=f)
            ds.text((L + (B - (cb[2] - cb[0])) / 2 - cb[0], cy0 - (cb[3] - cb[1]) / 2 - cb[1]),
                    text, font=f, fill=CLR_BC + (255,))
        rot = strip.rotate(-ang, expand=True, resample=Image.BICUBIC)
        cu = (gx0 + gx1) / 2.0 - L / 2.0                  # 沿閱讀方向：讓「框」對準印刷字
        cv = gy1 + th * 0.22 + sh / 2.0                   # 沿垂直方向：貼在字的正下方
        c = origin + r_hat * cu + d_hat * cv
        layer.alpha_composite(rot, (int(c[0] - rot.width / 2), int(c[1] - rot.height / 2)))
    out = Image.alpha_composite(base.convert("RGBA"), layer).convert("RGB")
    # 裁掉四周空白：原稿常常只有上半頁有內容，不裁的話印成一頁會小到看不清楚
    from PIL import ImageChops
    diff = ImageChops.difference(out, Image.new("RGB", out.size, (255, 255, 255))).convert("L")
    bb = diff.point(lambda v: 255 if v > 12 else 0).getbbox()
    if bb:
        m = int(max(out.width, out.height) * 0.012)
        out = out.crop((max(0, bb[0] - m), max(0, bb[1] - m),
                        min(out.width, bb[2] + m), min(out.height, bb[3] + m)))
    if out.width > maxw:
        out = out.resize((maxw, int(out.height * maxw / out.width)), Image.LANCZOS)
    # 整頁只有黑白紅綠幾個顏色，量化後檔案小很多，肉眼看不出差別
    return out.quantize(colors=64, method=Image.MEDIANCUT, dither=Image.NONE)


def load_pages(path):
    """逐頁產生 (page 或 None, 影像, scale)，避免一次渲染整份 PDF。"""
    if os.path.splitext(path)[1].lower() in (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"):
        from PIL import ImageOps, ImageSequence
        with Image.open(path) as source:
            for frame in ImageSequence.Iterator(source):
                img = ImageOps.exif_transpose(frame.copy()).convert("L")
                if img.width < 1600:
                    k = 1600 / img.width
                    img = img.resize((int(img.width * k), int(img.height * k)))
                yield None, img, 1.0
        return
    with fitz.open(path) as doc:
        if doc.needs_pass:
            raise ValueError("PDF 需要密碼，請先提供可讀取的檔案")
        for page in doc:
            yield (page,) + render(page, DPI)


def validate_expect(expect_total):
    if expect_total is not None and (type(expect_total) is not int or expect_total < 1):
        raise ValueError("應有條碼總數必須是大於 0 的整數")
    return expect_total


def check_pdf(path, expect_total=None):
    validate_expect(expect_total)
    rows, warns = [], []
    counted = False                      # 是否做得成數量核對
    for pno, (page, img, scale) in enumerate(load_pages(path)):
        W, H = img.size
        boxes = label_boxes(page, scale) if page is not None else []
        hi_cache = {}

        items, seen = [], []
        for r in zxingcpp.read_barcodes(img):
            q = quad_pts(r)
            c = q.mean(0)
            if any(abs(c[0] - s[0]) < 20 and abs(c[1] - s[1]) < 20 for s in seen):
                continue
            seen.append(c)
            items.append((r, q))
        items.sort(key=lambda t: (owner_label((t[1].min(0)[0], t[1].min(0)[1],
                                               t[1].max(0)[0], t[1].max(0)[1]), boxes),
                                  t[1].min(0)[1], t[1].min(0)[0]))

        if not items:
            warns.append("第 %d 頁：完全掃不到條碼，請確認是否為條碼稿、或線條太細。" % (pno + 1))

        for r, q in items:
            bbox = (q.min(0)[0], q.min(0)[1], q.max(0)[0], q.max(0)[1])
            treg, ang, skew = text_geom(q, W, H)
            row = dict(page=pno + 1, label=owner_label(bbox, boxes) + 1, fmt=str(r.format),
                       bc=norm(r.text), bc_raw=r.text, orient=int(ang) % 360, skew=skew,
                       ocr="", ocr2="", raw="", score=0.0, src="", shape="", shape_conf=0.0,
                       verdict="CHECK", reason="", glyphs=[])

            if treg is None or skew > MAX_SKEW:
                row["reason"] = "條碼傾斜 %.1f°，無法可靠對位" % skew
                row["img"] = to_b64(img.crop(tuple(int(v) for v in bbox)))
                rows.append(row)
                continue

            crop = upright(img.crop(tuple(int(v) for v in treg)), ang)

            # 文字層不能代替實際印字；隱藏文字、被覆蓋文字也會被抽取出來。
            wtext = read_pdf_words(page, treg, scale, ang) if page is not None else None
            row['pdf_text'] = norm(wtext) if wtext is not None else None
            t1, s1 = ocr_text(crop)
            if page is not None:
                if DPI_HI not in hi_cache:
                    hi_cache[DPI_HI] = render(page, DPI_HI)
                him, hscale = hi_cache[DPI_HI]
                k = hscale / scale
                crop_hi = upright(him.crop(tuple(int(v * k) for v in treg)), ang)
                hi_real = True          # PDF：800dpi 是真的重新渲染，資訊量確實增加
            else:
                crop_hi = crop.resize((crop.width * 2, crop.height * 2), Image.LANCZOS)
                hi_real = False         # 照片：放大只是內插，沒有新資訊
            t2, s2 = ocr_text(crop_hi)
            row.update(raw=t1, ocr=norm(t1), ocr2=norm(t2), score=min(s1, s2), src="OCR×2")
            row["glyphs"] = segment_glyphs(crop)
            row["_crop"] = crop
            row["_crop_hi"], row["_hi_real"] = crop_hi, hi_real
            # 顯示寬只有 240px，存 560px 已足夠放大看，不必存到 900px（報告會胖一倍）
            gb = row.get("glyphs") or []
            ext = ((min(g[0] for g in gb), max(g[0] + g[4] for g in gb),
                    min(g[3] for g in gb), max(g[3] + g[2] for g in gb)) if gb else None)
            row["_geom"] = (treg, ang, ext, [(g[0], g[4]) for g in gb])
            row["img"] = to_b64(upright(img.crop((min(bbox[0], treg[0]) - 10, min(bbox[1], treg[1]) - 10,
                                                  max(bbox[2], treg[2]) + 10, max(bbox[3], treg[3]) + 10)), ang),
                                maxw=560)
            rows.append(row)

        # 標籤間的數量落差——只能當「提示」，不能當數量已核對。
        # 每張標籤都漏讀同樣數量時，眾數會跟著一起錯，這個方法看不出來。
        if boxes:
            cnt = {}
            for r in rows:
                if r["page"] == pno + 1:
                    cnt[r["label"]] = cnt.get(r["label"], 0) + 1
            for i in range(len(boxes)):
                cnt.setdefault(i + 1, 0)
            vals = list(cnt.values())
            if vals:
                mode = max(set(vals), key=vals.count)
                for lb, v in sorted(cnt.items()):
                    if v != mode:
                        warns.append("提示：第 %d 頁第 %d 張標籤只掃到 %d 個條碼（其他標籤是 %d 個）——"
                                     "很可能有條碼印不出來或品質不良，請人工確認。" % (pno + 1, lb, v, mode))

    # 字形覆核
    if any(r.get("glyphs") for r in rows):
        nclu = shape_read(rows)
        print("  字形覆核：整份檔共 %d 種字形" % nclu)

    # 疊合對照圖（給人眼用的最後一道：不經 OCR 就能看出條碼字與印字對不對得上）
    for r in rows:
        r["cmp"] = ""
        r["tpl"], r["tpl_bad"], r["tpl_margin"] = "", [], 0.0
        r["tpl_lo"] = r["tpl_hi"] = ""
        r["tpl_filled"], r["tpl_res_conflict"] = [], []
        gl = r.get("glyphs") or []
        if gl and len(gl) == len(r["bc"]):
            # 兩個解析度都跑，不是「只在不方便時才重試」——否則 400dpi 讀錯但讀得很完整的列
            # 永遠不會被複查，等於只在對自己有利時才複查。成本可以忽略（每字 36 次內積）。
            tpl_lo, mg_lo = template_read(gl)
            tpl_hi, mg_hi = "", []
            gl_hi = segment_glyphs(r["_crop_hi"]) if r.get("_crop_hi") is not None else []
            if len(gl_hi) == len(r["bc"]):
                tpl_hi, mg_hi = template_read(gl_hi)
            n = len(tpl_lo)
            # 兩個解析度都讀出來、卻讀成不同的字 → 這本身就是警訊，不能挑一個用
            r["tpl_res_conflict"] = [i for i in range(min(n, len(tpl_hi)))
                                     if tpl_lo[i] != "?" and tpl_hi[i] != "?"
                                     and tpl_lo[i] != tpl_hi[i]]
            merged, margins, filled = [], [], []
            for i in range(n):
                c_lo = tpl_lo[i]
                c_hi = tpl_hi[i] if i < len(tpl_hi) else "?"
                if c_lo != "?":
                    merged.append(c_lo)
                    margins.append(mg_lo[i])
                elif c_hi != "?" and r.get("_hi_real"):
                    # 只有 PDF 的 800dpi 算「新證據」；照片放大是內插，不能拿來把 CHECK 升級
                    merged.append(c_hi)
                    margins.append(mg_hi[i])
                    filled.append(i)
                else:
                    merged.append("?")
                    margins.append(mg_lo[i])
            tpl = "".join(merged)
            r["tpl"], r["tpl_lo"], r["tpl_hi"] = tpl, tpl_lo, tpl_hi
            r["tpl_filled"] = filled
            r["tpl_margins"] = margins
            r["tpl_margin"] = min(margins) if margins else 0.0
            r["tpl_bad"] = [i for i, c in enumerate(tpl) if c != "?" and c != r["bc"][i]]
        try:
            if r.get("_crop") is not None and gl:
                im = compare_image(r["_crop"], r["glyphs"], r["bc"], r["ocr"],
                                   r.get("glyph_labels", []), r["tpl"], r["tpl_bad"])
                if im is not None:
                    # 這張圖只有黑/紅/藍/白幾個顏色，量化成 32 色可以省掉七成檔案大小
                    r["cmp"] = to_b64(im.quantize(colors=32, method=Image.MEDIANCUT,
                                                  dither=Image.NONE), maxw=960)
        except Exception as e:
            print("  疊合圖產生失敗（不影響判定）：%s" % e)
        r.pop("_crop", None)
        r.pop("_crop_hi", None)
        r.pop("glyphs", None)
        r.pop("glyph_labels", None)

    for r in rows:
        finalize(r)

    # 原圖對照頁（判定完才畫，才知道哪幾列要標紅）
    by_page = {}
    for r in rows:
        if r.get("_geom"):
            by_page.setdefault(r["page"], []).append(r)
    if by_page:
        try:
            for pno, (_pg, im, _sc) in enumerate(load_pages(path), 1):
                rs = by_page.get(pno)
                if not rs:
                    continue
                k = 0.5                      # 用一半解析度畫就夠，省記憶體
                small = im.resize((int(im.width * k), int(im.height * k)), Image.LANCZOS)
                geoms = [(tuple(v * k for v in r["_geom"][0]), r["_geom"][1],
                          tuple(v * k for v in r["_geom"][2]) if r["_geom"][2] else None,
                          [(x * k, w * k) for x, w in r["_geom"][3]],
                          r["bc"], "%d-%s" % (r["page"], r["label"] if r["label"] > 0 else "-"),
                          "ok" if r["verdict"] == "OK" else
                          ("ng" if r["verdict"] == "NG" else "check")) for r in rs]
                rs[0]["overview"] = to_b64(overview_image(small, geoms), maxw=1700)
        except Exception as e:
            print("  原圖對照頁產生失敗（不影響判定）：%s" % e)
    for r in rows:
        r.pop("_geom", None)

    # 同一張標籤上的多個條碼必須一致
    grp = {}
    for r in rows:
        grp.setdefault((r["page"], r["label"]), []).append(r)
    for (p, l), rs in grp.items():
        vals = sorted({x["bc"] for x in rs})
        if l > 0 and len(vals) > 1:
            warns.append("第 %d 頁第 %d 張標籤：同一張標籤上的條碼內容不一致（%s）。" % (p, l, "、".join(vals)))

    # 只有「外部給的應有總數」才算數量核對完成。
    # 程式自己從辨識結果推出來的數字不能拿來驗證自己。
    if expect_total is not None:
        if len(rows) != expect_total:
            warns.append("應有 %d 個條碼，實際只辨識到 %d 個——差額請人工確認。" % (expect_total, len(rows)))
        counted = True
    return rows, warns, counted


def summarize(rows, warns, counted):
    """算出整份檔的結論與燈號。放行只給「乾淨且總數已核對」。"""
    ng = [r for r in rows if r["verdict"] == "NG"]
    chk = [r for r in rows if r["verdict"] not in ("OK", "NG")]
    clean = bool(rows) and all(r["verdict"] == "OK" for r in rows) and not warns
    if clean and counted:
        return "全部一致，可以放行", "ok", len(ng), len(chk)
    if clean and not counted:
        return ("已辨識項目一致，但總數未核對——請填入應有條碼總數再確認一次", "warn", len(ng), len(chk))
    return "發現異常，先別開印", "bad", len(ng), len(chk)


def finalize(r):
    """定案。OK 只給有兩條互相獨立證據的列，其餘一律 CHECK 並寫明原因。"""
    if r["verdict"] == "CHECK" and r["reason"]:
        return                                   # 傾斜等前面已判定的情況
    bc, tx = r["bc"], r["ocr"]
    if not tx:
        r["verdict"], r["reason"] = "NOTEXT", "讀不到條碼下方的文字"
        return
    if r["src"] == "PDF文字層":
        r["verdict"], r["reason"] = "CHECK", "只有 PDF 文字層，尚未核對實際可見印字"
        return
    else:
        # 判定原則：OK/NG 要有「至少兩條互相印證的證據」；只有一條就叫人來看。
        # 但反過來說，某一軌稍有瑕疵、其他軌都吻合時不該叫人——那只會製造雜訊，
        # 看久了就會開始無視真正的警告。
        if not r.get('ocr2'):
            r['verdict'], r['reason'] = 'CHECK', '第二次 OCR 讀不到文字，無法完成雙次核對'
            return
        if r.get('pdf_text') is not None and r['pdf_text'] != tx:
            r['verdict'] = 'CHECK'
            r['reason'] = 'PDF 文字層與可見印字辨識不同（%s / %s），請看原圖確認' % (r['pdf_text'], tx)
            return
        if r["ocr2"] != tx:
            r["verdict"] = "CHECK"
            r["reason"] = "兩次獨立 OCR 讀出不同結果（%s / %s）" % (tx, r["ocr2"])
            return
        if r.get("tpl_res_conflict"):
            pos = "、".join(str(i + 1) for i in r["tpl_res_conflict"])
            r["verdict"] = "CHECK"
            r["reason"] = ("模板疊合比對在兩個解析度讀出不同的字（第 %s 字：400dpi=%s／800dpi=%s），"
                           "請看原圖確認" % (pos, r.get("tpl_lo", ""), r.get("tpl_hi", "")))
            r["text_conflict"] = True
            return
        sh, tp = r.get("shape", ""), r.get("tpl", "")
        for name, value in (("字形覆核", sh), ("模板疊合比對", tp)):
            if '?' in value and any(c != '?' and (i >= len(tx) or c != tx[i]) for i, c in enumerate(value)):
                r['verdict'] = 'CHECK'
                r['reason'] = '文字辨識有衝突——%s=%s；部分字雖未辨識，已確認的衝突仍須人工核對' % (name, value)
                r['text_conflict'] = True
                return
        sh_ok = bool(sh) and "?" not in sh
        tp_ok = bool(tp) and "?" not in tp
        tracks = [("OCR×2", tx)]
        if sh_ok:
            tracks.append(("字形覆核", sh))
        if tp_ok:
            tracks.append(("模板疊合比對", tp))          # 這條完全不經 OCR

        if len({v for _n, v in tracks}) > 1:              # 有軌互相矛盾
            r["verdict"] = "CHECK"
            r["reason"] = ("文字辨識有衝突——%s，實際印字待人工確認"
                           % "、".join("%s=%s" % (n, v) for n, v in tracks))
            r["text_conflict"] = True
            return

        names = [n for n, _v in tracks]
        if len(tracks) < 2:                               # 證據只有一條
            why = "字形覆核不可用（%s）" % ("未啟用 cv2" if not sh else "字數或票數不足")
            r["verdict"] = "CHECK"
            r["reason"] = "只有單軌 OCR：%s，也沒有模板疊合比對可佐證" % why
            return
        if r["score"] < MIN_SCORE and not tp_ok:          # 信心低又沒有不經 OCR 的佐證
            r["verdict"] = "CHECK"
            r["reason"] = ("OCR 信心 %.2f 低於門檻 %.2f，且沒有不經 OCR 的證據可佐證"
                           % (r["score"], MIN_SCORE))
            return
        if not tp_ok:
            miss = "、".join(str(i + 1) for i, c in enumerate(tp) if c == "?") if tp else ""
            r['verdict'] = 'CHECK'
            r['reason'] = ('模板疊合比對第 %s 字辨識不出（400dpi 與 800dpi 都試過）；'
                           '字形覆核票源仍是 OCR，不能當作獨立佐證' % miss) if miss else                           '模板疊合比對未完整辨識；字形覆核票源仍是 OCR，不能當作獨立佐證'
            return
        r["verdict"] = "OK" if tx == bc else "NG"
        low = ("；OCR 信心 %.2f 偏低，但有不經 OCR 的模板疊合比對佐證" % r["score"]
               if r["score"] < MIN_SCORE else "")
        # NG 這句要講清楚「一致的是三條軌彼此」，不是「跟條碼一致」——
        # 否則 NG 的列上寫著「全部一致」，一眼掃過去會誤以為沒事。
        agree = " 全部一致" if r["verdict"] == "OK" else " 讀出的印字一致，但與條碼不符"
        r["reason"] = "%s%s%s" % ("＋".join(names), agree, low)

    # 疊合比對的結果寫成人看得懂的一句話（判定本身已在上面用掉這條證據）
    tpl = r.get("tpl", "")
    if tpl:
        if "?" in tpl:
            r["tpl_note"] = "疊合比對有 %d 個字形狀分不出來，這幾個字沒被覆核到" % tpl.count("?")
            r["tpl_weak"] = True
        else:
            mg = r.get("tpl_margin", 0.0)
            ratio = mg / TPL_MIN_MARGIN if TPL_MIN_MARGIN else 0
            grade = "充裕" if ratio >= 5 else ("尚可" if ratio >= 2.5 else "偏低，建議看圖確認")
            r["tpl_weak"] = ratio < 2.5
            if r.get("tpl_bad"):
                mgs = r.get("tpl_margins") or []
                pairs = "，".join(
                    "第%d字印的像 %s、條碼卻是 %s（領先次像 %.2f）"
                    % (i + 1, tpl[i], bc[i], mgs[i] if i < len(mgs) else 0)
                    for i in r["tpl_bad"][:4])
                r["tpl_note"] = "疊合比對：%d 個字與條碼對不上——%s" % (len(r["tpl_bad"]), pairs)
            else:
                fill = ("；第 %s 字在 400dpi 判不出，改用 800dpi 解出"
                        % "、".join(str(i + 1) for i in r["tpl_filled"])) if r.get("tpl_filled") else ""
                r["tpl_note"] = ("疊合比對：%d 個字形狀全部吻合條碼（最難分辨的那個字，"
                                 "仍領先第二像的 %.2f＝門檻的 %.1f 倍，%s）%s"
                                 % (len(tpl), mg, ratio, grade, fill))
    if r["verdict"] == "OK" and not CODE_RE.match(bc):
        r["verdict"], r["reason"] = "FORMAT", "條碼與文字一致，但料號格式不符 9碼+A+8碼日期"


def build_html(path, rows, warns, counted):
    ng = [r for r in rows if r["verdict"] == "NG"]
    chk = [r for r in rows if r["verdict"] in ("CHECK", "NOTEXT", "FORMAT")]
    stat, tone, _n, _c = summarize(rows, warns, counted)
    color = {"ok": "#0a7d32", "warn": "#b26a00", "bad": "#c00"}[tone]
    codes = sorted({r["bc"] for r in rows})
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    badges = {"OK": ("OK", "#0a7d32"), "NG": ("NG 不一致", "#c00"),
              "NOTEXT": ("讀不到文字", "#b26a00"), "FORMAT": ("格式異常", "#b26a00"),
              "CHECK": ("需人工確認", "#b26a00")}

    def tplnote(r):
        note = r.get("tpl_note")
        if not note:
            return ""
        if r.get("tpl_bad") or "分不出來" in note:
            cls, mark = "tplbad", "⚠ "
        elif r.get("tpl_weak"):
            cls, mark = "tplweak", "△ "
        else:
            cls, mark = "tplok", "✓ "
        return '<div class="%s">%s%s</div>' % (cls, mark, escape(note))

    def cell(r):
        # 三個判讀值本來各佔一欄，整列會寬到超出畫面；併成一欄三行，比較起來也更直覺
        a, b = diff_html(r["bc"], r["ocr"])
        lb, col = badges[r["verdict"]]
        sh = r.get("shape") or "—"
        raw = (r.get("raw") or "").strip()
        vals = ('<div><span class="tag t-bc">條碼</span><span class="mono">%s</span></div>'
                '<div><span class="tag t-ocr">OCR</span><span class="mono">%s</span></div>'
                '<div><span class="tag t-sh">字形</span><span class="mono">%s</span>'
                '<span class="src">　%s</span></div>%s') % (
            a, b, escape(sh), ("%.0f%%" % (r["shape_conf"] * 100)) if r.get("shape") else "",
            ('<div class="src">原文：%s</div>' % escape(raw)) if raw and raw != r["ocr"] else "")
        return ("<tr class=\"%s\"><td>%d-%s</td><td><img src=\"%s\"></td><td>%s</td><td>%s</td>"
                "<td><span class=\"badge\" style=\"background:%s\">%s</span></td>"
                "<td class=\"src\">%s%s<br>%s　信心%.2f　%s / %d°</td></tr>") % (
            r["verdict"], r["page"], r["label"] if r["label"] > 0 else "-", r["img"], vals,
            ('<a href="%s" target="_blank" title="點開看原尺寸"><img class="cmp" src="%s"></a>'
             % (r["cmp"], r["cmp"])) if r.get("cmp") else "—",
            col, lb, escape(r["reason"]), tplnote(r), escape(r["src"]), r["score"], escape(r["fmt"]), r["orient"])

    warn_html = ("<ul>" + "".join("<li>%s</li>" % escape(w) for w in warns) + "</ul>") if warns else ""
    ng_lines, seen = [], set()
    for r in ng:
        key = (r["bc"], r["ocr"])
        if key in seen:
            continue
        seen.add(key)
        ng_lines.append("第%d頁 第%s張標籤：條碼掃出 %s ，但下方文字印 %s"
                        % (r["page"], r["label"] if r["label"] > 0 else "?", r["bc"], r["ocr"]))
    ng_box = ('<div class="ngbox"><h3>異常清單（可直接複製回報）</h3><pre>%s</pre></div>'
              % escape("\n".join(ng_lines))) if ng_lines else ""
    css = """
body{font-family:"Microsoft JhengHei",system-ui,sans-serif;margin:24px;color:#222;background:#fafafa;max-width:1600px}
h1{font-size:20px;margin:0 0 4px} .sub{color:#666;font-size:13px;margin-bottom:16px}
.stat{font-size:22px;font-weight:700;margin:12px 0 18px}
.tw{overflow-x:auto}
table{border-collapse:collapse;width:100%;min-width:980px;background:#fff}
th,td{border:1px solid #ddd;padding:8px;font-size:13px;vertical-align:middle}
th{background:#f0f0f0} img{max-width:240px;display:block} img.cmp{max-width:none;width:430px}
.tag{display:inline-block;width:34px;font-size:10.5px;color:#fff;border-radius:4px;text-align:center;
     margin-right:6px;vertical-align:1px}
.t-bc{background:#d60000} .t-ocr{background:#0052cc} .t-sh{background:#64748b}
.tplbad{color:#c00;font-weight:700;font-size:12px;margin-top:3px}
.tplok{color:#0a7d32;font-size:11px;margin-top:3px}
.tplweak{color:#b26a00;font-size:11.5px;margin-top:3px}
.ovh{font-size:17px;margin:22px 0 4px}
.ovblock{margin:10px 0 22px}
.ovbar{display:flex;gap:8px;align-items:center;margin:0 0 6px;font-size:12px;color:#666}
.ovbar button{background:#fff;border:1px solid #b6c2d1;color:#1f3a5f;border-radius:6px;
              padding:5px 11px;font-size:13px;cursor:pointer}
.ovbar button:hover{background:#eef4ff}
.ovbox{position:relative;width:100%;background:#fff;border:1px solid #ddd;border-radius:6px;overflow:hidden}
.ovbox img{position:absolute;left:50%;top:50%;transform-origin:center center;max-width:none;display:block}
.ngbox{background:#fff1f0;border:1px solid #ffa39e;border-radius:6px;padding:12px 14px;margin:0 0 16px}
.ngbox h3{margin:0 0 6px;font-size:14px}
.ngbox pre{margin:0;font-family:Consolas,monospace;font-size:13px;white-space:pre-wrap}
.mono{font-family:Consolas,monospace;font-size:15px;letter-spacing:1px;white-space:nowrap}
.bad{color:#c00;background:#ffe3e3;padding:0 2px;border-radius:3px}
.badge{color:#fff;padding:3px 8px;border-radius:10px;font-size:12px;white-space:nowrap}
tr.NG td{background:#fff5f5} tr.CHECK td,tr.NOTEXT td,tr.FORMAT td{background:#fffbe6}
.src{color:#777;font-size:11px} .sh{text-align:center;color:#334155}
ul{background:#fff7e6;border:1px solid #ffd591;padding:12px 12px 12px 30px;border-radius:6px}
.codes{font-family:Consolas,monospace;font-size:13px;color:#444}
.toolbar{margin:0 0 10px} .toolbar button{background:#1f3a5f;color:#fff;border:0;border-radius:6px;
         padding:7px 14px;font-size:13px;cursor:pointer}
@media print{
  @page{size:A4 landscape;margin:8mm}
  body{margin:0;background:#fff;max-width:none}
  .toolbar,.ovbar{display:none!important}
  .tw{overflow:visible} table{min-width:0}
  th,td{padding:5px;font-size:11px}
  td img{max-width:150px} img.cmp{width:290px!important}
  .mono{font-size:12px}
  tr,.ovblock,.ngbox{break-inside:avoid;page-break-inside:avoid}
  .ovbox{height:auto!important;position:static;overflow:visible;border:0}
  .ovbox img{position:static!important;transform:none!important;width:auto!important;
             max-width:100%!important;max-height:176mm!important;margin:0 auto}
  *{-webkit-print-color-adjust:exact;print-color-adjust:exact}
}
"""
    head = ('<meta charset="utf-8"><title>標籤條碼檢查報告</title><style>%s</style>'
            '<h1>標籤條碼檢查報告</h1>'
            '<div class="sub">檔案：%s　·　檢查時間：%s　·　共 %d 個條碼</div>'
            '<div class="toolbar"><button type="button" onclick="window.print()">🖨 列印 ／ 存成 PDF</button>'
            '<span class="sub">　（列印時會自動轉成橫式並縮排，按鈕不會印出來）</span></div>'
            '<div class="stat" style="color:%s">%s（不一致 %d 個，需確認 %d 個）</div>%s%s') % (
        css, escape(os.path.basename(path)), now, len(rows), color, stat, len(ng), len(chk), ng_box, warn_html)
    ovs = [(r["page"], r["overview"]) for r in rows if r.get("overview")]
    ov_html = ""
    if ovs:
        ov_html = ('<h2 class="ovh">原圖對照頁</h2>'
                   '<p class="sub">整頁原樣，紅字是<b>條碼解碼出來的內容</b>，貼在該標籤印刷數字的正下方，'
                   '方向跟著標籤本身走（原稿顛倒的，補上去的也跟著顛倒）。'
                   '<b>綠框＝已確認一致；紅框＋✗＝不一致或待確認，請看下方判定。</b>可以照著原本的版面逐張核對。</p>'
                   + "".join(
                       '<div class="ovblock"><div class="ovbar">'
                       '<button type="button" onclick="ovRot(this,-90)">↺ 左轉 90°</button>'
                       '<button type="button" onclick="ovRot(this,90)">↻ 右轉 90°</button>'
                       '<button type="button" onclick="ovRot(this,0,1)">回正</button>'
                       '<span>第 %d 頁　·　旋轉不影響判定，只是方便你照實際貼標方向核對</span>'
                       '</div><div class="ovbox"><img src="%s" onload="ovFit(this)"></div></div>'
                       % (pg, b64) for pg, b64 in ovs))
    table = ('<div class="tw"><table><tr><th>頁-標籤</th><th>條碼與文字</th>'
             '<th>三種判讀值<br><span class="src">條碼=解碼　OCR=辨識　字形=多實例覆核</span></th>'
             '<th>疊合對照<br><span class="src">紅=條碼字　藍=OCR字，疊回原印字上</span></th>'
             '<th>判定</th><th>依據</th></tr>%s</table></div>') % "".join(cell(r) for r in rows)
    foot = ('<p class="codes">本檔條碼解出的料號：%s</p>'
            '<p class="sub">說明：條碼內容由 zxing-cpp 解碼，Code 128 含檢查碼，讀不出來會回報「掃不到」而不是給錯值。'
            '下方可見文字以兩種解析度各辨識一次；PDF 文字層僅作輔助，不能代替畫面，'
            '兩次結果不同即標需人工確認，不會因為某一次剛好等於條碼就採用它。'
            '字形覆核是輔助手段——票源同樣來自 OCR，靠同一形狀的多個實例互相檢查，票數或得票比例不足時該列改標需人工確認。'
            'OK 需 OCR 結果與完整模板辨識一致。兩次 OCR 和字形覆核仍可能有共同誤差。判定非 OK 時，請以左側圖片目視確認後再回覆製作方。</p>'
            '<p class="sub"><b>疊合對照怎麼看：</b>紅字是<b>條碼解碼</b>出來的內容，藍字是 OCR 讀到的，'
            '兩者都逐字疊回原本印刷的字上（對位用的是字形切割，不經過 OCR）。'
            '<b>紅字和黑字對不上＝條碼與印字真的不一致</b>，這條完全不依賴 OCR，可以直接用肉眼確認。'
            '紅框標的是條碼與 OCR 判讀不同的位置，橘框標的是程式自己也不確定的字。</p>') % escape("、".join(codes))
    return head + "<script>\nfunction ovFit(img){\n  var r = +(img.dataset.rot || 0), box = img.parentElement;\n  var W = box.clientWidth - 16, nw = img.naturalWidth, nh = img.naturalHeight;\n  if (!nw) return;\n  var asp = nh / nw, w = (r % 180 === 0) ? Math.min(nw, W) : Math.min(nw, W / asp);\n  img.style.width = w + 'px';\n  box.style.height = (((r % 180 === 0) ? w * asp : w) + 16) + 'px';\n  img.style.transform = 'translate(-50%,-50%) rotate(' + r + 'deg)';\n}\nfunction ovRot(btn, d, reset){\n  var img = btn.closest('.ovblock').querySelector('img');\n  img.dataset.rot = reset ? 0 : ((+(img.dataset.rot || 0) + d) % 360 + 360) % 360;\n  ovFit(img);\n}\nwindow.addEventListener('resize', function(){\n  document.querySelectorAll('.ovbox img').forEach(ovFit);\n});\n</script>" + ov_html + table + foot


EDGE_PATHS = [
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
]


def find_edge():
    for p in EDGE_PATHS:
        if os.path.exists(p):
            return p
    import shutil
    return shutil.which("msedge") or shutil.which("chrome")


def html_to_pdf(html_path, pdf_path, timeout=150):
    """用 Windows 內建的 Edge 無視窗模式把報告轉成 PDF（不需額外套件，全程離線）。

    注意：Edge 的命令列參數若含非 ASCII 路徑會卡住不回應（報告檔名有中文，踩過），
    所以一律先複製到暫存區用純英文檔名轉，轉完再搬回原本的檔名。"""
    import subprocess, shutil, tempfile, time, urllib.request as _u
    edge = find_edge()
    if not edge:
        raise RuntimeError("找不到 Edge，無法轉 PDF")
    work = tempfile.mkdtemp(prefix="labelpdf_")
    try:
        tmp_html = os.path.join(work, "report.html")
        tmp_pdf = os.path.join(work, "report.pdf")
        shutil.copyfile(html_path, tmp_html)
        cmd = [edge, "--headless=new", "--disable-gpu", "--no-first-run",
               "--no-default-browser-check", "--user-data-dir=" + os.path.join(work, "prof"),
               "--no-pdf-header-footer", "--virtual-time-budget=15000",
               "--print-to-pdf=" + tmp_pdf, "file:" + _u.pathname2url(tmp_html)]
        deadline = time.monotonic() + timeout
        result = subprocess.run(cmd, timeout=timeout, capture_output=True,
                                creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        if result.returncode:
            detail = result.stderr.decode(errors='replace').strip()[-500:]
            raise RuntimeError("Edge 轉檔失敗（退出碼 %s）：%s" % (result.returncode, detail))
        # Edge's launcher can exit successfully before its background process writes
        # the PDF. Do not delete that process's working directory prematurely.
        previous_size = None
        while time.monotonic() < deadline:
            try:
                size = os.path.getsize(tmp_pdf)
                if size > 0 and size == previous_size:
                    with fitz.open(tmp_pdf) as result_doc:
                        if result_doc.page_count > 0:
                            break
                previous_size = size
            except (OSError, RuntimeError, ValueError):
                previous_size = None
            time.sleep(min(.25, max(0, deadline - time.monotonic())))
        else:
            raise RuntimeError("Edge 轉檔逾時，未取得完整 PDF（已等待 %s 秒）" % timeout)
        shutil.copyfile(tmp_pdf, pdf_path)
        return pdf_path
    finally:
        shutil.rmtree(work, ignore_errors=True)


def main():
    args = [a for a in sys.argv[1:]]
    expect = None
    want_pdf = "--pdf" in args
    if want_pdf:
        args.remove("--pdf")
    if "--expect" in args:
        i = args.index("--expect")
        try:
            expect = validate_expect(int(args[i + 1]))
            del args[i:i + 2]
        except (IndexError, ValueError):
            print("--expect 後面要接大於 0 的整數")
            return 2
    if not args:
        try:
            import tkinter as tk
            from tkinter import filedialog
            tk.Tk().withdraw()
            args = list(filedialog.askopenfilenames(
                title="選擇標籤 PDF 或照片",
                filetypes=[("標籤檔", "*.pdf;*.png;*.jpg;*.jpeg;*.tif;*.tiff"), ("所有檔案", "*.*")]))
        except Exception:
            pass
    if not args:
        print("沒有選擇檔案。")
        return 0
    bad = 0
    for path in args:
        print("檢查：" + path)
        rows, warns, counted = check_pdf(path, expect)
        out = os.path.splitext(path)[0] + "_條碼檢查報告.html"
        with open(out, "w", encoding="utf-8") as f:
            f.write(build_html(path, rows, warns, counted))
        marks = {"OK": "  OK", "NG": "  NG", "NOTEXT": "  ??", "FORMAT": "  !!", "CHECK": "  ?!"}
        for r in rows:
            print("%s  第%d頁-標籤%s  條碼=%s  文字=%s  %s" % (
                marks[r["verdict"]], r["page"], r["label"] if r["label"] > 0 else "?",
                r["bc"], r["ocr"], r["reason"]))
        for w in warns:
            print("  警告：" + w)
        n = sum(1 for r in rows if r["verdict"] != "OK")
        bad += n + len(warns) + (0 if counted else 1)
        print("→ %d 個條碼，非 OK %d 個，警告 %d 則%s" % (len(rows), n, len(warns), "" if counted else "，未核對總數"))
        print("→ 報告：" + out)
        if want_pdf:
            try:
                pdf = html_to_pdf(out, os.path.splitext(out)[0] + ".pdf")
                print("→ PDF：" + pdf)
            except Exception as e:
                print("→ PDF 轉檔失敗：%s" % e)
        try:
            webbrowser.open("file:///" + out.replace("\\", "/"))
        except Exception:
            pass
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
