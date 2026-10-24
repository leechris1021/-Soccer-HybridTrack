"""
评估脚本：对比 IoU 版 A-G 与 EIoU 版 A-G 的消融实验结果。
用法:
    python eval_10seq_trackeval.py
    python eval_10seq_trackeval.py --data_root <repo>/tracking/test/test --results_root <repo>/outputs/tracker_results/full49
"""
import os, shutil, argparse
import numpy as np
from pathlib import Path
import trackeval

PROJECT_ROOT = Path(__file__).resolve().parent          # = <repo>/scripts
REPO_ROOT = PROJECT_ROOT.parent                          # = <repo>

parser = argparse.ArgumentParser()
parser.add_argument('--data_root', type=str,
                    default=str(REPO_ROOT / 'tracking' / 'test' / 'test'),
                    help='SNMOT 数据集根目录（含 images/test/SNMOT-xxx，默认 <repo>/tracking/test/test）')
parser.add_argument('--results_root', type=str,
                    default=str(REPO_ROOT / 'outputs' / 'tracker_results' / 'full49'),
                    help='追踪结果根目录（含各实验子文件夹）')
args = parser.parse_args()

DATA_ROOT = args.data_root
RESULTS_ROOT = args.results_root
TEMP = str(REPO_ROOT / 'outputs' / 'tmp' / 'trackeval_10seq')

# 只评估前 10 个序列
seqs = ['SNMOT-116', 'SNMOT-117', 'SNMOT-118', 'SNMOT-119', 'SNMOT-120',
        'SNMOT-121', 'SNMOT-122', 'SNMOT-123', 'SNMOT-124', 'SNMOT-125']

# -----------------------------------------------------------------------
# 实验组定义
# key   : 显示名称（≤20字符，用作 TrackEval tracker 目录名）
# value : 结果 txt 所在子目录（相对 RESULTS_ROOT）
# -----------------------------------------------------------------------
# IoU 版 A-G
iou_experiments = {
    "A_iou_Baseline":     "A_baseline",
    "B_iou_ReID_RN50":    "ablation_noecc_noaflink_iou_resnet50",
    "C_iou_ReID_ECC":     "ablation_ecc_noaflink_iou_resnet50",
    "D_iou_ReID_AFLink":  "ablation_noecc_aflink_iou_resnet50",
    "E_iou_ALL_RN50":     "ablation_ecc_aflink_iou_resnet50",
    "F_iou_AFLink_OS":    "ablation_noecc_aflink_iou_osnet",
    "G_iou_ALL_OS":       "ablation_ecc_aflink_iou_osnet",
}

# EIoU 版 A-G
eiou_experiments = {
    "A_eiou_Baseline":    "A_baseline_eiou",
    "B_eiou_ReID_RN50":   "ablation_noecc_noaflink_eiou_resnet50",
    "C_eiou_ReID_ECC":    "ablation_ecc_noaflink_eiou_resnet50",
    "D_eiou_ReID_AFLink": "ablation_noecc_aflink_eiou_resnet50",
    "E_eiou_ALL_RN50":    "ablation_ecc_aflink_eiou_resnet50",
    "F_eiou_AFLink_OS":   "ablation_noecc_aflink_eiou_osnet",
    "G_eiou_ALL_OS":      "ablation_ecc_aflink_eiou_osnet",
}


def evaluate_experiment(tracker_name: str, folder: str) -> dict | None:
    """
    对单个实验调用 TrackEval，返回指标字典或 None（结果目录不存在/评估失败）。
    """
    full_folder = os.path.join(RESULTS_ROOT, folder)
    if not os.path.isdir(full_folder):
        return None

    # 构建临时目录结构
    if os.path.exists(TEMP):
        shutil.rmtree(TEMP)
    gt_base   = os.path.join(TEMP, 'gt', 'SNMOT-test')
    pred_base = os.path.join(TEMP, 'SNMOT-test', tracker_name, 'data')
    os.makedirs(pred_base, exist_ok=True)

    valid_seqs = []
    for seq in seqs:
        gt_src   = os.path.join(DATA_ROOT, 'images', 'test', seq, 'gt', 'gt.txt')
        ini_src  = os.path.join(DATA_ROOT, 'images', 'test', seq, 'seqinfo.ini')
        pred_src = os.path.join(full_folder, f"{seq}.txt")
        if not os.path.exists(gt_src):
            print(f"  [WARN] gt not found: {gt_src}")
            continue
        if not os.path.exists(pred_src):
            print(f"  [WARN] pred not found: {pred_src}")
            continue
        gt_dst = os.path.join(gt_base, seq, 'gt')
        os.makedirs(gt_dst, exist_ok=True)
        shutil.copy(gt_src,  os.path.join(gt_dst, 'gt.txt'))
        if os.path.exists(ini_src):
            shutil.copy(ini_src, os.path.join(gt_base, seq, 'seqinfo.ini'))
        shutil.copy(pred_src, os.path.join(pred_base, f"{seq}.txt"))
        valid_seqs.append(seq)

    if not valid_seqs:
        return None

    # 写 seqmap
    with open(os.path.join(TEMP, 'seqmap.txt'), 'w') as f:
        f.write('name\n')
        for s in valid_seqs:
            f.write(s + '\n')

    ec = trackeval.Evaluator.get_default_eval_config()
    ec.update({'PRINT_RESULTS': False, 'PRINT_CONFIG': False,
               'DISPLAY_LESS_PROGRESS': True})

    dc = trackeval.datasets.MotChallenge2DBox.get_default_dataset_config()
    dc.update({
        'GT_FOLDER':         os.path.join(TEMP, 'gt'),
        'TRACKERS_FOLDER':   TEMP,
        'BENCHMARK':         'SNMOT',
        'SPLIT_TO_EVAL':     'test',
        'TRACKERS_TO_EVAL':  [tracker_name],
        'SEQMAP_FILE':       os.path.join(TEMP, 'seqmap.txt'),
        'DO_PREPROC':        False,
        'PRINT_CONFIG':      False,
    })
    mc = {'METRICS': ['HOTA', 'CLEAR', 'Identity'], 'THRESHOLD': 0.5}

    try:
        r, _ = trackeval.Evaluator(ec).evaluate(
            [trackeval.datasets.MotChallenge2DBox(dc)],
            [trackeval.metrics.HOTA(mc), trackeval.metrics.CLEAR(mc), trackeval.metrics.Identity(mc)])
        c = r['MotChallenge2DBox'][tracker_name]['COMBINED_SEQ']['pedestrian']
        return {
            'HOTA':  float(np.mean(c['HOTA']['HOTA'])),
            'DetA':  float(np.mean(c['HOTA']['DetA'])),
            'AssA':  float(np.mean(c['HOTA']['AssA'])),
            'MOTA':  float(c['CLEAR']['MOTA']),
            'IDF1':  float(c['Identity']['IDF1']),
            'IDSw':  int(c['CLEAR']['IDSW']),
        }
    except Exception as e:
        import traceback
        print(f"  [评估错误] {tracker_name}: {e}")
        traceback.print_exc(file=__import__('sys').stdout)
        return None


def print_table(title: str, experiments: dict, results: dict):
    print(f"\n{'='*85}")
    print(f"  {title}")
    print(f"{'='*85}")
    print(f"{'实验':<28}  {'HOTA':>7}  {'DetA':>7}  {'AssA':>7}  {'MOTA':>7}  {'IDF1':>7}  {'IDSw':>6}")
    print(f"{'-'*85}")
    for name in experiments:
        r = results.get(name)
        if r:
            print(f"{name:<28}  {r['HOTA']:>7.4f}  {r['DetA']:>7.4f}  {r['AssA']:>7.4f}  {r['MOTA']:>7.4f}  {r['IDF1']:>7.4f}  {r['IDSw']:>6d}")
        else:
            print(f"{name:<28}  {'N/A':>7}  {'N/A':>7}  {'N/A':>7}  {'N/A':>7}  {'N/A':>7}  {'N/A':>6}")
    print(f"{'='*85}")


def print_diff_table(iou_res: dict, eiou_res: dict, iou_exp: dict, eiou_exp: dict):
    """并排打印 IoU vs EIoU，并显示 HOTA/DetA/AssA 差值"""
    labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G']
    iou_keys  = list(iou_exp.keys())
    eiou_keys = list(eiou_exp.keys())

    W = 175
    print(f"\n{'='*W}")
    print(f"  IoU vs EIoU 对比（差值 = EIoU - IoU）")
    print(f"{'='*W}")
    print(f"{'实验':<3}  {'IoU HOTA':>9}  {'IoU DetA':>9}  {'IoU AssA':>9}  {'IoU MOTA':>9}  {'IoU IDF1':>9}  {'IoU IDSw':>8}  "
          f"{'EIoU HOTA':>10}  {'EIoU DetA':>10}  {'EIoU AssA':>10}  {'EIoU MOTA':>10}  {'EIoU IDF1':>10}  {'EIoU IDSw':>9}  "
          f"{'ΔHOTA':>7}  {'ΔDetA':>7}  {'ΔAssA':>7}  {'ΔMOTA':>7}  {'ΔIDF1':>7}  {'ΔIDSw':>7}")
    print(f"{'-'*W}")
    for i, lbl in enumerate(labels):
        ri = iou_res.get(iou_keys[i])
        re = eiou_res.get(eiou_keys[i])
        if ri and re:
            dh = re['HOTA'] - ri['HOTA']
            dd = re['DetA'] - ri['DetA']
            da = re['AssA'] - ri['AssA']
            dm = re['MOTA'] - ri['MOTA']
            df = re['IDF1'] - ri['IDF1']
            ds = re['IDSw'] - ri['IDSw']
            sh = '+' if dh >= 0 else ''
            sd = '+' if dd >= 0 else ''
            sa = '+' if da >= 0 else ''
            sm = '+' if dm >= 0 else ''
            sf = '+' if df >= 0 else ''
            ss = '+' if ds >= 0 else ''
            print(f"{lbl:<3}  {ri['HOTA']:>9.4f}  {ri['DetA']:>9.4f}  {ri['AssA']:>9.4f}  {ri['MOTA']:>9.4f}  {ri['IDF1']:>9.4f}  {ri['IDSw']:>8d}  "
                  f"{re['HOTA']:>10.4f}  {re['DetA']:>10.4f}  {re['AssA']:>10.4f}  {re['MOTA']:>10.4f}  {re['IDF1']:>10.4f}  {re['IDSw']:>9d}  "
                  f"{sh}{dh:>6.4f}  {sd}{dd:>6.4f}  {sa}{da:>6.4f}  {sm}{dm:>6.4f}  {sf}{df:>6.4f}  {ss}{ds:>6d}")
        else:
            print(f"{lbl:<3}  {'N/A':>9}  {'N/A':>9}  {'N/A':>9}  {'N/A':>9}  {'N/A':>9}  {'N/A':>8}  "
                  f"{'N/A':>10}  {'N/A':>10}  {'N/A':>10}  {'N/A':>10}  {'N/A':>10}  {'N/A':>9}  "
                  f"{'N/A':>7}  {'N/A':>7}  {'N/A':>7}  {'N/A':>7}  {'N/A':>7}  {'N/A':>7}")
    print(f"{'='*W}")


# -----------------------------------------------------------------------
# 主评估流程
# -----------------------------------------------------------------------
print("开始评估 IoU 组 A-G ...")
iou_results = {}
for tracker_name, folder in iou_experiments.items():
    print(f"  评估: {tracker_name} ...", end=' ', flush=True)
    r = evaluate_experiment(tracker_name, folder)
    iou_results[tracker_name] = r
    if r:
        print(f"HOTA={r['HOTA']:.4f}  AssA={r['AssA']:.4f}  IDF1={r['IDF1']:.4f}  IDSw={r['IDSw']}")
    else:
        print("跳过（结果目录不存在）")

print("\n开始评估 EIoU 组 A-G ...")
eiou_results = {}
for tracker_name, folder in eiou_experiments.items():
    print(f"  评估: {tracker_name} ...", end=' ', flush=True)
    r = evaluate_experiment(tracker_name, folder)
    eiou_results[tracker_name] = r
    if r:
        print(f"HOTA={r['HOTA']:.4f}  AssA={r['AssA']:.4f}  IDF1={r['IDF1']:.4f}  IDSw={r['IDSw']}")
    else:
        print("跳过（结果目录不存在）")

# 打印完整表格
print_table("IoU 版消融实验结果", iou_experiments, iou_results)
print_table("EIoU 版消融实验结果", eiou_experiments, eiou_results)
print_diff_table(iou_results, eiou_results, iou_experiments, eiou_experiments)
