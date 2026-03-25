# scripts/list_chb_seizures.py
import os, re, json
def get_seizure_times_from_summary(summary_path, file_name):
    seizure_times=[]
    with open(summary_path,'r') as f:
        lines=f.readlines()
    in_section=False; start=None
    for L in lines:
        if f"File Name: {file_name}" in L:
            in_section=True; start=None; continue
        if not in_section: continue
        m1 = re.search(r"Seizure Start Time: (\d+) seconds", L)
        m2 = re.search(r"Seizure End Time: (\d+) seconds", L)
        if m1: start=int(m1.group(1))
        if m2 and start is not None:
            seizure_times.append((start,int(m2.group(1))))
            start=None
        if "File Name:" in L and file_name not in L:
            break
    return seizure_times

base = r"E:\RESEARCH\DATABASE\CHB-MIT"   # change if needed
patients = sorted([d for d in os.listdir(base) if d.startswith("chb")])
summary = {}
for p in patients:
    pdir = os.path.join(base,p)
    sums = [f for f in os.listdir(pdir) if f.endswith("-summary.txt")]
    if not sums: continue
    sfile = os.path.join(pdir, sums[0])
    total_intervals = 0
    for f in os.listdir(pdir):
        if f.endswith(".edf"):
            st = get_seizure_times_from_summary(sfile, f)
            if st:
                total_intervals += len(st)
    summary[p] = total_intervals
print(json.dumps(summary, indent=2))
