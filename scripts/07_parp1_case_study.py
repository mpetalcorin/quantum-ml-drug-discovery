#!/usr/bin/env python
from pathlib import Path

from qmldd.parp1 import make_parp1_report

text = make_parp1_report()
out = Path("results/parp1_case_study.md")
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(text, encoding="utf-8")
print(text)
