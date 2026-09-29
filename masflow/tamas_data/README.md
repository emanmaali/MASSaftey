# TAMAS data (vendored)

Real data files copied verbatim from `github.com/microsoft/TAMAS`
(Kavathekar et al., "TAMAS: Benchmarking Adversarial Risks in
Multi-Agent LLM Systems", arXiv:2511.05269), for use by
`masflow/tamas_dpi_tasks.py` and `masflow/baseline_tamas_dpi.py`
per `experimental_protocol.md` §12/§13. Not our own dataset — see
TAMAS_LICENSE (code, MIT) and TAMAS_LICENSE.CDLA-2.0 (data).

- `*_DPI.json`: the 5 domain files (education/finance/healthcare/legal/
  news) from `data/DPI/`, 10 instances each, 50 total.
- `tools/`: the 20 per-agent tool schema files from
  `data/tools/autogen/`, used to build realistic tool-call system
  prompts for the target agent in each DPI instance.
