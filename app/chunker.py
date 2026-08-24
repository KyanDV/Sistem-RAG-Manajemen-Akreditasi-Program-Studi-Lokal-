import os
import re
import pymupdf
from app.config import CHUNK_TARGET_SIZE, COURSE_FILENAME_MAP

ROW_THRESHOLD = 6
TABLE_MIN_ROWS = 3
TABLE_MIN_COLS = 3
COLUMN_CLUSTER_GAP = 50
COLUMN_ASSIGN_TOLERANCE = 60
HEADING_PATTERNS = [
    re.compile(r'^\d+\.\s+[A-Z]'),
    re.compile(r'^[A-Z][A-Z\.\s/]{5,}'),
    re.compile(r'^[A-Z][a-z]+.*:$'),
    re.compile(r'^(BAB|BAGIAN|LAMPIRAN)\s+\d+'),
]
SECTION_MARKERS = [
    "pustaka", "media pembelajaran", "perangkat lunak", "perangkat keras",
    "dosen pengampu", "matakuliah syarat", "rencana pembelajaran",
    "mg ke-", "dicetak oleh", "deskripsi singkat mk",
]


def _extract_course_code(filename: str) -> str:
    name = filename.replace(".pdf", "").replace("Salinan ", "").strip()
    match = re.search(r'(IF|TK|SI)\d{5}', name)
    if match:
        return match.group(0)
    name_lower = name.lower()
    for pattern, code in COURSE_FILENAME_MAP.items():
        if pattern in name_lower:
            return code
    return ""


def _is_table_row(blocks_in_row):
    if len(blocks_in_row) < TABLE_MIN_COLS:
        return False
    x0s = sorted(set(b[1] for b in blocks_in_row))
    clusters = [[x0s[0]]]
    for x in x0s[1:]:
        if x - clusters[-1][-1] < COLUMN_CLUSTER_GAP:
            clusters[-1].append(x)
        else:
            clusters.append([x])
    return len(clusters) >= TABLE_MIN_COLS


def _format_table(rows):
    x0s = sorted(set(b[1] for row in rows for b in row))
    clusters = [[x0s[0]]]
    for x in x0s[1:]:
        if x - clusters[-1][-1] < COLUMN_CLUSTER_GAP:
            clusters[-1].append(x)
        else:
            clusters.append([x])
    columns = []
    for cluster in clusters:
        related = [b for b in [bb for row in rows for bb in row] if b[1] in cluster]
        cx0 = min(b[1] for b in related)
        cx1 = max(b[3] for b in related)
        columns.append((cx0, cx1))
    lines = []
    for row in rows:
        cells = ["" for _ in columns]
        for b in row:
            best_i = 0
            best_dist = abs(b[1] - columns[0][0])
            for i in range(1, len(columns)):
                dist = abs(b[1] - columns[i][0])
                if dist < best_dist:
                    best_dist = dist
                    best_i = i
            text = b[0].replace("\n", " ")
            if cells[best_i]:
                cells[best_i] += " " + text
            else:
                cells[best_i] = text
        lines.append(" | ".join(cells))
    return "\n".join(lines)


def _is_heading(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return False
    for pat in HEADING_PATTERNS:
        if pat.search(stripped):
            return True
    if len(stripped) < 60 and stripped.endswith(":") and not stripped.startswith("http"):
        return True
    return False


def _is_section_marker(text: str) -> bool:
    tl = text.strip().lower()
    return any(tl.startswith(m) or tl == m for m in SECTION_MARKERS)


def _merge_chunks(chunks, target_size=CHUNK_TARGET_SIZE):
    merged = []
    current = ""
    for chunk in chunks:
        if not current:
            current = chunk
        elif len(current) + len(chunk) < target_size:
            current += "\n" + chunk
        else:
            merged.append(current)
            current = chunk
    if current:
        merged.append(current)
    return merged


def extract_chunks(pdf_path: str):
    doc = pymupdf.open(pdf_path)
    fname = os.path.basename(pdf_path)
    course_code = _extract_course_code(fname)

    chunks = []
    chunk_pages = []
    chunk_meta = []

    for page_num, page in enumerate(doc, 1):
        blocks = page.get_text("blocks")
        if not blocks:
            continue

        sorted_blocks = sorted(blocks, key=lambda b: (b[1], b[0]))

        table_rows = []
        flow_lines = []
        i = 0

        while i < len(sorted_blocks):
            b = sorted_blocks[i]
            text = b[4].strip()
            if not text:
                i += 1
                continue

            row_blocks = [(text, b[0], b[1], b[2], b[3])]
            y0 = b[1]
            j = i + 1
            while j < len(sorted_blocks):
                nb = sorted_blocks[j]
                ntext = nb[4].strip()
                if not ntext:
                    j += 1
                    continue
                if abs(nb[1] - y0) <= ROW_THRESHOLD:
                    row_blocks.append((ntext, nb[0], nb[1], nb[2], nb[3]))
                    j += 1
                else:
                    break
            i = j

            if _is_table_row(row_blocks):
                table_rows.append(row_blocks)
            else:
                if table_rows and len(table_rows) >= TABLE_MIN_ROWS:
                    table_text = _format_table(table_rows)
                    header = f"[TABLE page={page_num}]"
                    chunks.append(f"{header} {table_text}")
                    chunk_pages.append(page_num)
                    chunk_meta.append({
                        "source": fname,
                        "course_code": course_code,
                        "page_number": page_num,
                        "chunk_type": "table",
                    })
                    table_rows = []
                for block in row_blocks:
                    flow_lines.append(block[0])

        if table_rows and len(table_rows) >= TABLE_MIN_ROWS:
            table_text = _format_table(table_rows)
            header = f"[TABLE page={page_num}]"
            chunks.append(f"{header} {table_text}")
            chunk_pages.append(page_num)
            chunk_meta.append({
                "source": fname,
                "course_code": course_code,
                "page_number": page_num,
                "chunk_type": "table",
            })
            table_rows = []

        text_chunks = []
        current = ""
        for line in flow_lines:
            stripped = line.strip()
            if not stripped:
                if current:
                    text_chunks.append(current)
                    current = ""
                continue
            if _is_heading(stripped) or _is_section_marker(stripped):
                if current:
                    text_chunks.append(current)
                current = stripped
            elif len(current) + len(stripped) < CHUNK_TARGET_SIZE:
                current = (current + "\n" + stripped) if current else stripped
            else:
                text_chunks.append(current)
                current = stripped
        if current:
            text_chunks.append(current)

        text_chunks = _merge_chunks(text_chunks)
        for tc in text_chunks:
            header = f"[{fname}]"
            chunks.append(f"{header} {tc}")
            chunk_pages.append(page_num)
            chunk_meta.append({
                "source": fname,
                "course_code": course_code,
                "page_number": page_num,
                "chunk_type": "text",
            })

    doc.close()
    return chunks, chunk_pages, course_code, chunk_meta
