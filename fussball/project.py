from __future__ import annotations

import os
from pathlib import Path

EXPECTED_TEST_SEQUENCE_COUNT = 49


def _looks_like_project_root(candidate: Path) -> bool:
    return (candidate / 'fussball').is_dir() and (candidate / 'scripts').is_dir()


def resolve_project_root() -> Path:
    env_candidates = [
        os.environ.get('FUSSBALL_ROOT'),
        os.environ.get('AUTODL_FUSSBALL_ROOT'),
        os.environ.get('AUTODL_PROJECT_ROOT'),
    ]
    for raw in env_candidates:
        if not raw:
            continue
        candidate = Path(raw).expanduser().resolve()
        if _looks_like_project_root(candidate):
            return candidate

    current = Path(__file__).resolve()
    for candidate in [current.parent, *current.parents]:
        if _looks_like_project_root(candidate):
            return candidate

    for candidate in (Path('/root/autodl-tmp/Fussball'), Path('/autodl-tmp/Fussball')):
        if _looks_like_project_root(candidate):
            return candidate.resolve()

    return current.parents[1]


PROJECT_ROOT = resolve_project_root()
CONFIGS_ROOT = PROJECT_ROOT / 'configs'
DATA_ROOT = PROJECT_ROOT / 'data'
REPORTS_ROOT = PROJECT_ROOT / 'reports'
WEIGHTS_ROOT = PROJECT_ROOT / 'weights'
OUTPUTS_ROOT = PROJECT_ROOT / 'outputs'
TRACKER_RESULTS_ROOT = OUTPUTS_ROOT / 'tracker_results'
TOOLS_ROOT = PROJECT_ROOT / 'tools'
THIRD_PARTY_ROOT = PROJECT_ROOT / 'third_party'

DEFAULT_TRACKING_ROOT = DATA_ROOT / 'tracking'
DEFAULT_TEST_DATA_ROOT = DEFAULT_TRACKING_ROOT / 'test' / 'test'
DEFAULT_TRAIN_DATA_ROOT = DEFAULT_TRACKING_ROOT / 'train' / 'train'
DEFAULT_CHALLENGE_DATA_ROOT = DEFAULT_TRACKING_ROOT / 'challenge' / 'challenge'
DEFAULT_TRACKING_2023_ROOT = DATA_ROOT / 'tracking-2023'

DEFAULT_YOLO_WEIGHTS = WEIGHTS_ROOT / 'best.pt'
DEFAULT_OSNET_WEIGHTS = WEIGHTS_ROOT / 'osnet_x075_soccernet_best.pth'
DEFAULT_AFLINK_WEIGHTS = WEIGHTS_ROOT / 'aflink_model_v3.pth'
DEFAULT_RAW_PRED_CACHE = OUTPUTS_ROOT / 'raw_pred_cache.json'

DEFAULT_SEQMAP_CANDIDATES = [
    TOOLS_ROOT / 'SNMOT-test.txt',
    THIRD_PARTY_ROOT / 'sn-tracking-main' / 'tools' / 'SNMOT-test.txt',
]


def load_default_test_sequence_names(gt_root: Path | None = None) -> list[str]:
    for seqmap_path in DEFAULT_SEQMAP_CANDIDATES:
        if seqmap_path.exists():
            seqs = [
                line.strip()
                for line in seqmap_path.read_text(encoding='utf-8').splitlines()
                if line.strip() and line.strip().lower() != 'name'
            ]
            if len(seqs) != EXPECTED_TEST_SEQUENCE_COUNT:
                raise RuntimeError(
                    f'SNMOT 测试序列清单应为 {EXPECTED_TEST_SEQUENCE_COUNT} 个，'
                    f'但在 {seqmap_path} 中读取到 {len(seqs)} 个。'
                )
            return seqs

    if gt_root is not None and gt_root.exists():
        seqs = sorted(
            p.name
            for p in gt_root.iterdir()
            if p.is_dir() and (p / 'gt' / 'gt.txt').exists() and (p / 'seqinfo.ini').exists()
        )
        if len(seqs) == EXPECTED_TEST_SEQUENCE_COUNT:
            return seqs

    raise FileNotFoundError(
        '未找到有效的 49 序列测试清单，请检查 `tools/SNMOT-test.txt`。'
    )
