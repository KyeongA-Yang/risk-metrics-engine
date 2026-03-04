# Learning Notes

## 2026-03-03 — Project setup / tooling basics
- `pyenv`로 Python 버전을 관리하고, 프로젝트별로 `.venv` 가상환경을 만들어 패키지 충돌을 피했다.
- `pip install -e .` + `pyproject.toml`로 `src/` 기반 패키지를 정석 방식으로 설치해 `PYTHONPATH` 없이 import 되게 했다.
- 프로젝트 구조를 `src/`(라이브러리), `tests/`(pytest), `scripts/`(실행 스크립트), `notebooks/`(분석 노트), `data/`(샘플 데이터)로 나눴다.
- CLI는 `argparse`로 `--csv`, `--alpha`, `--col` 옵션을 받아 재현 가능한 실행을 가능하게 했다.
- `pytest`로 자동 테스트를 만들고, Git(`git status/add/commit/push`)로 변경 이력을 관리했다.
- 리스크 기본 정의: `loss = -pnl`, `VaR_α = q_α(loss)`, `ES_α = E[loss | loss ≥ VaR_α]`.

## 2026-03-04 — Pandas time series / rolling / alignment
- `pd.to_datetime()`은 날짜/시간을 **datetime 타입**으로 변환한다.
- 시계열 연산 전 정석: `set_index("date").sort_index()`로 날짜 인덱스를 만들고 정렬한다.
- `rolling(window=3)`은 “최근 3개 관측치” 기준으로 계산하며 초반 `window-1` 구간은 `NaN`이 생긴다.
- `dropna()`는 `NaN`이 있는 행을 제거해 계산 가능한 구간만 남긴다(너무 일찍 쓰면 길이/인덱스가 달라질 수 있음).
- pandas Series 연산은 인덱스 기준으로 자동 정렬(alignment)되므로, 비교 전 `pd.concat([...], axis=1).dropna()`로 공통 인덱스를 맞추면 안전하다.
