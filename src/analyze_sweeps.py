import csv
import os
import sys
import scipy.io
import sweeps as s

data_dir = "../data/"
mat_ext = ".mat"


if __name__ == "__main__":
    data_subdir = len(sys.argv) > 1 and sys.argv[1] or ""
    fns = [
        f.split(".mat")[0]
        for f in os.listdir(f"../data/{data_subdir}")
        if f.endswith(mat_ext)
    ]
    if not os.path.exists(f"../results/{data_subdir}"):
        os.makedirs(f"../results/{data_subdir}")

    if not data_subdir.endswith("/"):
        data_subdir += "/"

    headers = [
        "file_id",
        "sweep",
        "artifact_number",
        "sanity_check",
        "n1_muV",
        "N1_nVsec",
        "N1prop",
        "n2_muV",
        "N2_nVsec",
        "N2prop",
        "pre_artifact_baseline_mV",
        "artifact_msec",
        "maximum_0_msec",
        "maximum_0_muV",
        "minimum_0_msec",
        "minimum_0_muV",
        "maximum_1_msec",
        "maximum_1_muV",
        "minimum_1_msec",
        "minimum_1_muV",
    ]
    cfn = f"../results/{data_subdir[:-1]}.csv"
    csv_file = open(cfn, "w")
    csv_handle = csv.writer(csv_file)
    csv_handle.writerow(headers)
    for fn in fns:
        ifn = data_dir + data_subdir + fn + mat_ext
        print(ifn)
        matlabdata = scipy.io.loadmat(ifn)
        _sweeps = []
        for idx, sweep_data in enumerate(matlabdata["data"]):
            sweep = s.sweep(sweep_data, "%s_%i" % (data_subdir + fn, idx))
            res = sweep.get_data_array(fn, idx)
            if res is not None:
                for r in res:
                    csv_handle.writerow(r)
            sweep.plot("%s_%i" % (data_subdir + fn, idx))
            sweep.plot_artifact()
            _sweeps.append(sweep)

    csv_file.close()

    print("Test successful. All done.")
