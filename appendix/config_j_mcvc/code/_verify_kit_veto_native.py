"""方案8 植入 AFLink 后的等价性验证。

对照物 = 旧诊断子类（KitVetoLinker）当年跑出的既定结果，随仓库发布在
  `_verify_kit_veto_native/PIX_c25_out_t25m2uS/`（HOTA=64.122，否决候选 2872）

三处等价性：
  A) 否决候选数 == 2872（候选级口径，完全决定输出）
  B) postprocessed/*.txt 与既定目录逐字节一致
  C) 走「JSON 落盘 + 序列名索引」的生产路径（--kit_labels）仍逐字节一致
外加两个自检：off 必须零否决、all（破坏性）必须远超常规否决量。
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parent                        # config_j_mcvc/code
ROOT = Path(os.environ.get('FUSSBALL_ROOT', CODE_DIR.parents[2]))  # 仓库根
sys.path.insert(0, str(CODE_DIR))
sys.path.insert(0, str(ROOT))

import aflink_hybrid_eval as AHE  # noqa: E402
import _exp_residual_gate as EX  # noqa: E402
import _probe_color_team as P  # noqa: E402

KEY = 'PIX_c25_out_t25m2uS'
# 既定结果（HOTA 64.122）参考包：随仓库发布的 `_verify_kit_veto_native/<KEY>/`
REF = Path(os.environ.get('KIT_VETO_REF_DIR', ROOT / 'appendix' / 'config_j_mcvc' / '_verify_kit_veto_native' / KEY))
# 本次运行的输出目录（仓库内可写位置，不污染参考包）
WORK = Path(os.environ.get('KIT_VETO_OUT_ROOT',
                           ROOT / 'appendix' / 'config_j_mcvc' / 'out' / '_verify_kit_veto_native'))
RECORDED_VETOED = 2872


def same_dir(ref: Path, got: Path) -> tuple[int, list[str], list[str]]:
    ref_files = sorted(f.name for f in ref.glob('*.txt'))
    bad = [n for n in ref_files if not (got / n).exists() or (got / n).read_bytes() != (ref / n).read_bytes()]
    extra = sorted(f.name for f in got.glob('*.txt') if f.name not in set(ref_files))
    return len(ref_files), bad, extra


def main() -> None:
    seqs = AHE.load_default_test_sequence_names(EX.GT)
    preds = AHE.load_mot_predictions(EX.RESULTS, seqs)
    cache = P.load_color_cache(preds, seqs, 8, False)
    pix = P.pixel_predict(cache, seqs, 'abL', 0.25, 2)
    lookup = P.build_pixel_lookup(preds, seqs, cache, 0.25, 0.30, feature='abL', pix=pix,
                                  veto_outliers=True, trim_frac=0.25, max_trim=2,
                                  out_conf=None, out_unlab=True, out_strict=True)

    # A + B：脚本内直接注入（id(track_map) 键）
    lk = P.make_veto_linker(lookup, 'any')
    processed, _ = AHE.apply_aflink(preds, lk)
    print('A) 原生 kit_veto 累计:', lk.kit_veto_totals)
    assert lk.kit_veto_totals['vetoed'] == RECORDED_VETOED, \
        f"否决候选 {lk.kit_veto_totals['vetoed']} != 既定 {RECORDED_VETOED}"

    ref = REF
    out = WORK / KEY
    AHE.write_predictions_to_dir(processed, out)
    n, bad, extra = same_dir(ref, out)
    print(f'B) 逐字节对比 {n} 个文件：不一致 {len(bad)}，多出 {len(extra)}')
    assert not bad and not extra, f'不一致 {bad[:5]} / 多出 {extra[:5]}'

    # C：JSON 落盘 + 序列名索引（生产入口口径）
    lab_path = WORK / 'labels.json'
    P.dump_kit_labels(preds, seqs, cache, lab_path, feature='abL', th=0.25, sil_min=0.30,
                      veto_outliers=True, trim_frac=0.25, max_trim=2,
                      out_conf=None, out_unlab=True, out_strict=True)
    lk2 = AHE.HybridAFLink(**EX.linker_kwargs(rule_variant='adaptive_cone', cone_g0=EX.CONE_G0),
                           kit_veto='any', kit_lookup=AHE.load_kit_labels(lab_path))
    processed2, _ = AHE.apply_aflink(preds, lk2)
    print('C) 生产路径(JSON) 累计:', lk2.kit_veto_totals)
    assert lk2.kit_veto_totals == lk.kit_veto_totals, 'JSON 生产路径否决量与脚本注入口径不同'
    out2 = WORK / f'{KEY}_json'
    AHE.write_predictions_to_dir(processed2, out2)
    n2, bad2, extra2 = same_dir(ref, out2)
    print(f'C) 逐字节对比 {n2} 个文件：不一致 {len(bad2)}，多出 {len(extra2)}')
    assert not bad2 and not extra2, f'不一致 {bad2[:5]} / 多出 {extra2[:5]}'

    # 自检：off 零否决；all 全否
    lk_off = P.make_veto_linker(lookup, 'off')
    AHE.apply_aflink(preds, lk_off)
    assert lk_off.kit_veto_totals['vetoed'] == 0, lk_off.kit_veto_totals
    lk_all = P.make_veto_linker(lookup, 'all')
    AHE.apply_aflink(preds, lk_all)
    print('自检) off 否决 0 | all 否决', lk_all.kit_veto_totals)
    assert lk_all.kit_veto_totals['vetoed'] > lk.kit_veto_totals['vetoed']

    print('\nOK：方案8 已由 AFLink 原生实现，且与既定 64.122 结果逐位一致。')


if __name__ == '__main__':
    main()
