
from __future__ import annotations

from pathlib import Path
import pandas as pd


def write_tables(tables: dict[str, pd.DataFrame], output_dir: Path) -> None:
    output_dir = Path(output_dir)
    csv_dir = output_dir / "csv"
    csv_dir.mkdir(parents=True, exist_ok=True)
    for name, df in tables.items():
        if isinstance(df, pd.DataFrame):
            df.to_csv(csv_dir / f"{name}.csv", index=False)


def safe_sheet_name(name: str, existing: set[str]) -> str:
    clean = "".join(c if c not in r'[]:*?/\\' else "_" for c in name)
    clean = clean[:31] if len(clean) > 31 else clean
    base = clean or "Sheet"
    out = base
    i = 1
    while out in existing:
        suffix = f"_{i}"
        out = base[:31-len(suffix)] + suffix
        i += 1
    existing.add(out)
    return out


def write_excel_workbook(tables: dict[str, pd.DataFrame], path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(path, engine="xlsxwriter") as writer:
        used = set()
        for name, df in tables.items():
            if not isinstance(df, pd.DataFrame):
                continue
            sheet = safe_sheet_name(name, used)
            df.to_excel(writer, index=False, sheet_name=sheet)
            worksheet = writer.sheets[sheet]
            workbook = writer.book
            header_fmt = workbook.add_format({"bold": True, "text_wrap": True, "valign": "top", "border": 1})
            for col_idx, col in enumerate(df.columns):
                worksheet.write(0, col_idx, col, header_fmt)
                max_len = max([len(str(col))] + [len(str(x)) for x in df[col].head(200).fillna("").tolist()])
                worksheet.set_column(col_idx, col_idx, min(max(max_len + 2, 10), 42))
            worksheet.freeze_panes(1, 0)
            if len(df) > 0 and len(df.columns) > 0:
                worksheet.autofilter(0, 0, len(df), len(df.columns) - 1)
