import json
from typing import List, Optional, Dict

import uvicorn
import pymorphy2
import re
from fastapi import FastAPI, Request, Depends
from pydantic import BaseModel
from contextlib import asynccontextmanager

class LawLink(BaseModel):
    law_id: Optional[int] = None
    article: Optional[str] = None
    point_article: Optional[str] = None
    subpoint_article: Optional[str] = None

class LinksResponse(BaseModel):
    links: List[LawLink]

class TextRequest(BaseModel):
    text: str
    
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    with open("law_aliases.json", "r") as file:
        codex_aliases = json.load(file)
    app.state.codex_aliases = codex_aliases
    print("🚀 Сервис запускается...")
    yield
    # Shutdown
    del codex_aliases
    print("🛑 Сервис завершается...")

def get_codex_aliases(request: Request) -> Dict:
    return request.app.state.codex_aliases

app = FastAPI(
    title="Law Links Service",
    description="Cервис для выделения юридических ссылок из текста",
    version="1.0.0",
    lifespan=lifespan
)

@app.post("/detect")
async def get_law_links(
    data: TextRequest,
    request: Request,
    codex_aliases: Dict = Depends(get_codex_aliases),
) -> LinksResponse:
    import re
    from typing import Optional, List

    text = data.text

    # ===== кэш морфо и алиасов =====
    global morph, alias_exact_map, alias_lemma_map, alias_max_len, _alias_ready
    try:
        _ = _alias_ready
    except NameError:
        _alias_ready = False

    if not _alias_ready:
        import pymorphy2
        morph = pymorphy2.MorphAnalyzer()
        alias_exact_map = {}
        alias_lemma_map = {}
        alias_max_len = 0

        def _lem_seq(s: str) -> List[str]:
            words = re.findall(r"[A-Za-zА-Яа-яЁё]+", s.lower())
            return [(w if w == "рф" else morph.parse(w)[0].normal_form) for w in words]

        for law_id_str, names in codex_aliases.items():
            lid = int(law_id_str)
            for name in names:
                key = name.lower().strip()
                if not key:
                    continue
                alias_exact_map[key] = lid

                lemmas = _lem_seq(key)
                if not lemmas:
                    continue

                # полная лемма-последовательность
                alias_lemma_map[tuple(lemmas)] = lid
                alias_max_len = max(alias_max_len, len(lemmas))

                # добавим все смежные подфразы длиной >= 2
                if len(lemmas) >= 2:
                    for L in range(len(lemmas), 1, -1):
                        for i in range(0, len(lemmas) - L + 1):
                            sub = tuple(lemmas[i:i+L])
                            alias_lemma_map.setdefault(sub, lid)
        _alias_ready = True

    # ===== утилиты =====
    pattern_abbrev = re.compile(r"\bст\.(?=\s*\d)", re.IGNORECASE)
    pattern_word = re.compile(r"\bстат(?:ья|ьи|ье|ьей|ью|ей|ьям|ьями|ьях)\b(?=\s*\d)", re.IGNORECASE)
    occurrences = [(m.start(), 'abbrev') for m in pattern_abbrev.finditer(text)]
    occurrences += [(m.start(), 'word')   for m in pattern_word.finditer(text)]
    occurrences.sort()

    def find_first_digit_pos(s: str, start_idx: int) -> Optional[int]:
        m = re.search(r"\d", s[start_idx:])
        return start_idx + m.start() if m else None

    # читаем «список статей» до начала названия закона;
    # поддерживаем 'и', 'или', 'и/или' внутри списка
    def extract_article_tokens(s: str, start_idx: Optional[int]) -> (str, int):
        if start_idx is None:
            return "", start_idx or 0
        i = start_idx
        while i < len(s) and s[i].isspace():
            i += 1
        out = []
        while i < len(s):
            ch = s[i]
            if ch.isdigit() or ch in ".,- ":
                out.append(ch)
                i += 1
                continue
            if ch.isalpha():
                j = i
                while j < len(s) and s[j].isalpha():
                    j += 1
                word = s[i:j].lower()
                if word in ("и", "или", "и/или"):
                    out.append(",")
                    i = j
                    while i < len(s) and s[i].isspace():
                        i += 1
                    continue
                # любое иное слово — начало названия закона
                break
            break
        return "".join(out).strip(), i

    # нормализация элемента (фикс «оборудования.ъ» -> «ъ»)
    def _last_alnum(fragment: str) -> str:
        parts = re.findall(r"[A-Za-zА-Яа-яЁё0-9]+", fragment)
        return parts[-1] if parts else fragment.strip()

    def parse_range_list(token_str: str, kind: Optional[str] = None) -> List[str]:
        token_str = (token_str or '').strip()
        if not token_str:
            return []
        token_str = re.sub(r"(?<=\w)\s+(?:и/или|или|и)\s+(?=\w)", ",", token_str, flags=re.IGNORECASE)
        parts_raw = [t.strip() for t in token_str.split(",") if t.strip()]
        result: List[str] = []
        for raw in parts_raw:
            if "-" in raw and kind in {"sub", "point", "part"}:
                a_raw, b_raw = [x.strip() for x in raw.split("-", 1)]
                a, b = _last_alnum(a_raw), _last_alnum(b_raw)
                if a.isdigit() and b.isdigit():
                    ai, bi = int(a), int(b)
                    if 0 <= bi - ai <= 1000:
                        result.extend([str(n) for n in range(ai, bi + 1)])
                        continue
                if len(a) == 1 and len(b) == 1 and a.isalpha() and b.isalpha():
                    step = 1 if ord(a) <= ord(b) else -1
                    result.extend([chr(c) for c in range(ord(a), ord(b) + step, step)])
                    continue
                result.append(b if b else a)
                continue

            tok = raw
            if kind in {"sub", "point", "part"}:
                tok = re.sub(r"^(?i)(?:и/или|или|и)\s+", "", tok).strip()
                tok = _last_alnum(tok)

            if kind == "article":
                result.append(tok)
            else:
                if re.fullmatch(r"[A-Za-zА-Яа-яЁё]", tok) or tok.isdigit():
                    result.append(tok)
                else:
                    tok2 = _last_alnum(tok)
                    if tok2 and (re.fullmatch(r"[A-Za-zА-Яа-яЁё]", tok2) or tok2.isdigit()):
                        result.append(tok2)
        return result

    def tokenize_law_snippet(snippet: str) -> List[str]:
        return re.findall(r"[A-Za-zА-Яа-яЁё№\-]+", snippet)

    def resolve_law_id(law_tokens: List[str]) -> Optional[int]:
        candidate = " ".join(law_tokens).strip().lower()
        if candidate in alias_exact_map:
            return alias_exact_map[candidate]

        words = re.findall(r"[A-Za-zА-Яа-яЁё]+", candidate)
        cand_lemmas = [(w if w == "рф" else morph.parse(w)[0].normal_form) for w in words]
        if not cand_lemmas:
            return None

        # ищем любой самый длинный подсовпадающий n-грам в карте
        Lmax = min(alias_max_len, len(cand_lemmas))
        for L in range(Lmax, 1, -1):
            for i in range(0, len(cand_lemmas) - L + 1):
                window = tuple(cand_lemmas[i:i+L])
                lid = alias_lemma_map.get(window)
                if lid is not None:
                    return lid
        return None

    # левый контекст (маркеры -> свои списки)
    marker_re = re.compile(r"(подпп\.|пп\.|подпункт\w*|п\.|пункт\w*|ч\.|част\w*)", re.IGNORECASE)

    def parse_reference_parts(left_segment: str) -> (List[str], List[str], List[str]):
        subs: List[str] = []
        points: List[str] = []
        parts: List[str] = []
        markers = list(marker_re.finditer(left_segment))
        for i, m in enumerate(markers):
            kind_word = m.group(1).lower()
            start = m.end()
            end = markers[i+1].start() if i + 1 < len(markers) else len(left_segment)
            chunk = left_segment[start:end].strip(" \t\r\n,.;:—-–—()[]{}«»")
            if not chunk:
                continue
            if kind_word.startswith("подпп") or kind_word == "пп." or kind_word.startswith("подпункт"):
                subs.extend(parse_range_list(chunk, kind="sub"))
            elif kind_word == "п." or kind_word.startswith("пункт"):
                points.extend(parse_range_list(chunk, kind="point"))
            elif kind_word == "ч." or kind_word.startswith("част"):
                parts.extend(parse_range_list(chunk, kind="part"))
        return subs, points, parts

    links: List[LawLink] = []
    last_law_id: Optional[int] = None

    for idx, _ in occurrences:
        # 1) статьи (включая «…, 18 и 1287.2»)
        digit_pos = find_first_digit_pos(text, idx)
        article_tokens_str, law_start_idx = extract_article_tokens(text, digit_pos)
        article_list = parse_range_list(article_tokens_str, kind="article")

        # 2) левый контекст
        left_segment = text[max(0, idx - 200): idx]
        sub_list, point_list, part_list = parse_reference_parts(left_segment)

        # 3) правый контекст (название закона)
        law_snippet_end = len(text)
        for sep in [".", "!", "?", ",", ";", "\n", ")", "—", " - ", " -", "- ", "…", "..."]:
            pos = text.find(sep, law_start_idx)
            if pos != -1:
                law_snippet_end = min(law_snippet_end, pos)
        law_snippet = text[law_start_idx: law_snippet_end]
        law_tokens = tokenize_law_snippet(law_snippet)

        law_id = last_law_id if "того же" in law_snippet.lower() else resolve_law_id(law_tokens)
        if law_id is not None:
            last_law_id = law_id

        # 4) комбинации часть/пункт
        if part_list and point_list:
            combined_points = [f"{prt}.{pt}" for prt in part_list for pt in point_list]
        elif part_list:
            combined_points = part_list[:]
        elif point_list:
            combined_points = point_list[:]
        else:
            combined_points = [None]

        # 5) формирование ссылок
        if not sub_list and combined_points == [None]:
            for art in article_list:
                links.append(LawLink(law_id=law_id, article=str(art)))
            continue

        for art in article_list:
            if sub_list:
                pt0 = combined_points[0] if combined_points else None
                for sub in sub_list:
                    links.append(LawLink(
                        law_id=law_id,
                        article=str(art),
                        point_article=str(pt0) if pt0 is not None else None,
                        subpoint_article=str(sub),
                    ))
                if len(combined_points) > 1:
                    for extra_pt in combined_points[1:]:
                        links.append(LawLink(law_id=law_id, article=str(art), point_article=str(extra_pt)))
            else:
                for pt in combined_points:
                    links.append(LawLink(law_id=law_id, article=str(art), point_article=str(pt) if pt is not None else None))

    return LinksResponse(links=links)


@app.get("/health")
async def health_check():
    """
    Проверка состояния сервиса
    """
    return {"status": "healthy"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8978)
