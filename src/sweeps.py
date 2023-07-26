import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate


class sweep(object):
    # This object handles the analysis of a sweep:
    # - interface with the main routine to receive data
    # - detect the threshold
    # - instantiate the artifact_analysis (aa) class
    # - report back the n1, n2, N1, N2, N1prop, and N2prop
    # - generate plots: full sweep done here, individual artifacts via aa class
    def __init__(self, data, tag=""):
        self.tag = tag  # contains subdirectory, filname, and sweep id.
        self._measuring_frequency_kHz = 50  # in kHz
        # self._smoothing_window = 7  # in data points
        self._smoothing_window = 5  # in data points
        self._artifact_dead_space = 5  # in milliseconds
        # self._artifact_threshold = -8000  # in microVolts
        self._artifact_threshold = -1500  # in microVolts
        self._res_dir = "../results/"
        self._png_ext = ".png"
        self._data = np.array(data.copy())
        self._x_coords = np.arange(len(self._data)) / self._measuring_frequency_kHz
        self._smooth_data = self.smoothen(self._smoothing_window)
        self._artifact_analyses = []
        self._analyze()

    def smoothen(self, window):
        return np.convolve(self._data, np.ones(window), "same") / window

    def _analyze(self):
        self._artifacts_mask = self._data < self._artifact_threshold
        for i in range(1, len(self._artifacts_mask)):
            if self._artifacts_mask[i - 1] and self._artifacts_mask[i]:
                self._artifacts_mask[i - 1] = False
        self._artifacts_x = self._x_coords[self._artifacts_mask].copy()
        delete_indices = []
        for idx in range(1, len(self._artifacts_x)):
            if (
                self._artifacts_x[idx] - self._artifacts_x[idx - 1]
                < self._artifact_dead_space
            ):
                delete_indices.append(idx - 1)
        self._artifacts_x = np.delete(self._artifacts_x, delete_indices)

        self.data_analysis = []
        for artifact_number in range(len(self._artifacts_x)):
            artifact_x = self._artifacts_x[artifact_number]
            artifact_idx = np.where(self._x_coords == artifact_x)[0][0]
            aa = artifact_analysis(
                self.tag,
                artifact_number,
                self._artifact_threshold,
                self._x_coords,
                self._data,
                self._smooth_data,
                artifact_idx,
                self._measuring_frequency_kHz,
            )
            self._artifact_analyses.append(aa)

    def get_data_array(self, tag=None, idx=None):
        data_array = []
        N1 = None
        N2 = None
        for artifact_number in range(len(self._artifact_analyses)):
            self._artifact_analyses[artifact_number].set_previous_N(N1, N2)
            aa_i = self._artifact_analyses[artifact_number].get_data_array(idx)
            data_array.append(aa_i)
            N1 = self._artifact_analyses[artifact_number].get_N1()
            N2 = self._artifact_analyses[artifact_number].get_N2()
        return data_array

    def plot(self, tag):
        # Plot the whole sweep in one plot
        # if len(self._artifacts_x) == 0:
        #     # print(f"{tag} : No artifacts found. Not generating plot().")
        #     return
        # plt.clf()
        fig = plt.figure(figsize=(20, 4))
        for artifact_x in self._artifacts_x:
            plt.axvline(x=artifact_x, color="y", linewidth=1.0)

        plt.plot(self._x_coords, self._data, ".", markersize=1.5)
        plt.plot(self._x_coords, self._smooth_data, "k-", linewidth=0.5)
        plt.axhline(
            y=self._artifact_threshold,
            color="m",
            linestyle="-",
            linewidth=1.0,
            label="artifact threshold",
        )
        plt.title(tag)
        plt.xlim(min(self._x_coords), max(self._x_coords))
        plt.xlabel("Time [ms]")
        plt.ylabel("Amplitude [$\mu$V]")
        fig.tight_layout()
        plt.savefig(f"{self._res_dir}{tag}___overview{self._png_ext}", dpi=600)
        plt.close()

    def plot_artifact(self):
        # Plot a single artifact per plot.
        if len(self._artifacts_x) == 0:
            return
        for artifact_number in range(len(self._artifacts_x)):
            self._artifact_analyses[artifact_number].plot_artifact()


class artifact_analysis(object):
    # This object contains all of the relevant information and tools
    # pertaining to the analysis a single artifact.
    def __init__(
        self,
        tag,
        artifact_number,
        artifact_threshold,
        time_stamps,
        raw_data,
        smooth_data,
        artifact_idx,
        measuring_frequency_kHz,
    ):
        self.series = tag.split("/")[0]
        self.tag = tag.split("/")[1]
        self.artifact_number = artifact_number
        self._artifact_threshold = artifact_threshold
        self._res_dir = "../results/"
        self._png_ext = ".png"
        self._smoothing_window = 7  # in data points
        self._measuring_frequency_kHz = measuring_frequency_kHz
        self._pre_artifact_capture = 1  # in msec
        self._post_artifact_capture = 5  # in msec
        self._first_max_cutoff = 0.5  # in msec: max time between artifact and first max
        self._focus_window = int(
            self._post_artifact_capture * self._measuring_frequency_kHz
        )
        self.di = int((self._smoothing_window - 1) / 2)  # Delta index due to smoothing

        # baseline: points before artifact used for reference
        self._baseline_x = time_stamps[artifact_idx - 30 : artifact_idx - 15]
        self._baseline_y = raw_data[artifact_idx - 30 : artifact_idx - 15]
        self.baseline = sum(self._baseline_y) / len(self._baseline_y)

        # artifact padding: measurements after artifact being ignored as invalid measurements
        self._artifact_padding = self.di
        for i in range(1, int(self._measuring_frequency_kHz / 2)):
            if raw_data[artifact_idx + i] > self.baseline + 100:
                self._artifact_padding = self.di + i + 1

        # artifact index: index (of internal data array) where artifact is detected
        self._artifact_idx = min(
            self._measuring_frequency_kHz * self._pre_artifact_capture, artifact_idx
        )
        self._time_stamps = time_stamps[
            artifact_idx - self._artifact_idx : artifact_idx + self._focus_window
        ]
        self._data = raw_data[
            artifact_idx - self._artifact_idx : artifact_idx + self._focus_window
        ]
        self._smooth_data = smooth_data[
            artifact_idx - self._artifact_idx : artifact_idx + self._focus_window
        ]
        self._focus_id0 = self._artifact_idx + self._artifact_padding
        self._compute_minmax()
        self._perform_pre_check()
        self.set_previous_N(None, None)
        self._analyze_artifact()

    def focus_x(self):
        return self._time_stamps[self._focus_id0 :]

    def focus_y(self):
        return self._data[self._focus_id0 :]

    def sfocus_y(self):
        return self._smooth_data[self._focus_id0 :]

    def _compute_minmax(self):
        extremum_detector = np.gradient(self.sfocus_y())
        maxima_ids_raw = []
        for i in range(0, len(extremum_detector) - 1):
            if extremum_detector[i] > 0 and extremum_detector[i + 1] <= 0:
                if np.abs(extremum_detector[i]) <= np.abs(extremum_detector[i + 1]):
                    maxima_ids_raw.append(i)
                else:
                    maxima_ids_raw.append(i + 1)
        self.maxima_ids = maxima_ids_raw
        # self.maxima_ids = []
        # for idx in maxima_ids_raw:
        #     d = self.focus_y()[max(idx - self.di, 0) : idx + self.di + 1]
        #     q = np.where(d == max(d))[0][0]
        #     if (
        #         self.focus_y()[q + max(idx - self.di - 3, 0)]
        #         <= self.focus_y()[q + max(idx - self.di, 0)]
        #     ):
        #         self.maxima_ids.append(q + max(idx - self.di, 0))
        if len(self.maxima_ids) > 0:
            starting_point = self.maxima_ids[0]
        else:
            starting_point = 0
        self.minima_ids = []
        for i in range(starting_point, len(extremum_detector) - 1):
            if extremum_detector[i] < 0 and extremum_detector[i + 1] > 0:
                if np.abs(extremum_detector[i]) <= np.abs(extremum_detector[i + 1]):
                    self.minima_ids.append(i)
                else:
                    self.minima_ids.append(i + 1)

    def _perform_pre_check(self):
        # Initial data integrity checks:
        # - do we detect more than one minimum and maximum?
        # - if so, do we detect a maximum before the first minimum?
        # - and do we detect the first maximum within the cutoff time?
        self._pre_check = False
        if (len(self.minima_ids) > 1) and (len(self.maxima_ids) > 1):
            if (self.maxima_ids[0] < self.minima_ids[0]) and (
                (self._artifact_padding + self.maxima_ids[0])
                / self._measuring_frequency_kHz
                < self._first_max_cutoff
            ):
                self._pre_check = True

        if len(self.maxima_ids) < 2:
            print(
                f"{self.tag} ERROR: cannot find at 2 maxima; only finding {len(self.maxima_ids)}. {self.maxima_ids}. Check input data"
            )
        if len(self.minima_ids) < 2:
            print(
                f"{self.tag} ERROR: cannot find at 2 minima; only finding {len(self.minima_ids)}. Check input data"
            )

    def set_previous_N(self, previousN1=None, previousN2=None):
        self._previousN1 = previousN1
        self._previousN2 = previousN2

    def get_N1(self):
        return self._N1

    def get_N2(self):
        return self._N2

    def get_N1prop(self):
        if self._N1 is not None and self._previousN1 is not None:
            return self._N1 / self._previousN1
        else:
            return None

    def get_N2prop(self):
        if self._N2 is not None and self._previousN2 is not None:
            return self._N2 / self._previousN2
        else:
            return None

    def _get_slope(self, ax, ay, bx, by):
        return (by - ay) / (bx - ax)

    def _get_slope_point(self, ax, ay, bx, by, cx):
        return ay + (by - ay) / (bx - ax) * (cx - ax)

    def _analyze_artifact(self):
        self._tid1 = None  # index at which first tangent is detected
        self._t1x = None  # x (i.e. time) value at first tangent
        self._t1y = None  # y (i.e. muV) value at first tangent
        self._c1y = None  # y value on line connecting max and tangent at n1
        self._n1 = None  # maximum amplitude 1
        self._N1 = None  # area 1
        self._tid2 = None  # index at which second tangent is detected
        self._t2x = None  # x (i.e. time) value at second tangent
        self._t2y = None  # y (i.e. muV) value at second tangent
        self._c2y = None  # y value on line connecting max and tangent at n2
        self._n2 = None  # maximum amplitude 2
        self._N2 = None  # area 2
        self.sanity_check = False
        if self._pre_check:
            focus_x = self.focus_x()
            focus_y = self.focus_y()
            sfocus_y = self.sfocus_y()
            # XXX PERFORMING MEASUREMENTS ON SMOOTHENED FUNCTION XXX
            # focus_y = sfocus_y.copy()
            # XXX ---------------------------------------------- XXX

            dx = 1 / self._measuring_frequency_kHz
            tan1_search_start = max(self.maxima_ids[0] + 10, self.minima_ids[0])
            f1x = focus_x[tan1_search_start:]
            f1y = sfocus_y[tan1_search_start:]
            tangent_finder_1 = (
                self._get_slope(
                    focus_x[self.maxima_ids[0]], focus_y[self.maxima_ids[0]], f1x, f1y
                )
                - np.gradient(f1y) * self._measuring_frequency_kHz
            )
            tid_array_1 = np.where(np.diff(np.sign(tangent_finder_1)))[0]
            if len(tid_array_1) > 0:
                self._tid1 = min(tan1_search_start + tid_array_1[0], self.maxima_ids[1])
                self._t1x = focus_x[self._tid1]
                # self._t1y = focus_y[self._tid1]
                self._t1y = sfocus_y[self._tid1]
                # c1yarray = self._get_slope_point(
                #     focus_x[self.maxima_ids[0]],
                #     focus_y[self.maxima_ids[0]],
                #     self._t1x,
                #     self._t1y,
                #     focus_x[self.maxima_ids[0] : self._tid1],
                # )
                c1yarray = self._get_slope_point(
                    focus_x[self.maxima_ids[0]],
                    sfocus_y[self.maxima_ids[0]],
                    self._t1x,
                    self._t1y,
                    focus_x[self.maxima_ids[0] : self._tid1],
                )
                # n1array = c1yarray - focus_y[self.maxima_ids[0] : self._tid1]
                n1array = c1yarray - sfocus_y[self.maxima_ids[0] : self._tid1]
                self._n1 = max(n1array)
                c1i = np.where(n1array == self._n1)[0][0]
                self.minima_ids[0] = self.maxima_ids[0] + c1i
                self._c1y = c1yarray[c1i]
            # self._N1 = 0.5 * (
            #     focus_x[self.maxima_ids[1]] - focus_x[self.maxima_ids[0]]
            # ) * (focus_y[self.maxima_ids[0]] + focus_y[self.maxima_ids[1]]) - dx * sum(
            #     focus_y[self.maxima_ids[0] : self.maxima_ids[1]]
            # )
            self._N1 = 0.5 * (
                focus_x[self.maxima_ids[1]] - focus_x[self.maxima_ids[0]]
            ) * (
                sfocus_y[self.maxima_ids[0]] + sfocus_y[self.maxima_ids[1]]
            ) - dx * sum(
                sfocus_y[self.maxima_ids[0] : self.maxima_ids[1]]
            )
            tan2_search_start = max(
                self.maxima_ids[1] + int(0.2 * self._measuring_frequency_kHz),
                self.minima_ids[1],
            )
            f2x = focus_x[tan2_search_start:]
            f2y = sfocus_y[tan2_search_start:]
            tangent_finder = (
                self._get_slope(
                    focus_x[self.maxima_ids[1]], focus_y[self.maxima_ids[1]], f2x, f2y
                )
                - np.gradient(f2y) * self._measuring_frequency_kHz
            )
            tid_array_2 = np.where(np.diff(np.sign(tangent_finder)))[0]
            if len(tid_array_2) > 0:
                tid_array_2_clean = [
                    i
                    for i in tid_array_2
                    if focus_y[tan2_search_start + i] - focus_y[self.maxima_ids[1]] > 10
                ]
                self._tid2 = tan2_search_start + tid_array_2_clean[0]
                # self._tid2 = tan2_search_start + tid_array_2[0]
                self._t2x = focus_x[self._tid2]
                # self._t2y = focus_y[self._tid2]
                self._t2y = sfocus_y[self._tid2]
                # c2yarray = self._get_slope_point(
                #     focus_x[self.maxima_ids[1]],
                #     focus_y[self.maxima_ids[1]],
                #     self._t2x,
                #     self._t2y,
                #     focus_x[self.maxima_ids[1] : self._tid2],
                # )
                c2yarray = self._get_slope_point(
                    focus_x[self.maxima_ids[1]],
                    sfocus_y[self.maxima_ids[1]],
                    self._t2x,
                    self._t2y,
                    focus_x[self.maxima_ids[1] : self._tid2],
                )
                # n2array = c2yarray - focus_y[self.maxima_ids[1] : self._tid2]
                n2array = c2yarray - sfocus_y[self.maxima_ids[1] : self._tid2]
                self._n2 = max(n2array)
                c2i = np.where(n2array == self._n2)[0][0]
                # self.minima_ids[1] = self.maxima_ids[1]+c2i
                self.minima_ids[1] = self.maxima_ids[1] + c2i
                self._c2y = c2yarray[c2i]
                # self._N2 = 0.5 * (self._t2x - focus_x[self.maxima_ids[1]]) * (
                #     focus_y[self.maxima_ids[1]] + self._t2y
                # ) - dx * sum(focus_y[self.maxima_ids[1] : self._tid2])
                self._N2 = 0.5 * (self._t2x - focus_x[self.maxima_ids[1]]) * (
                    sfocus_y[self.maxima_ids[1]] + self._t2y
                ) - dx * sum(sfocus_y[self.maxima_ids[1] : self._tid2])
                self.sanity_check = True
        self.sanity_check = self.sanity_check and None not in [
            self._tid1,
            self._t1x,
            self._t1y,
            self._c1y,
            self._n1,
            self._N1,
            self._tid2,
            self._t2x,
            self._t2y,
            self._c2y,
            self._n2,
            self._N2,
        ]

    def get_data_array(self, idx=None):
        current_analysis = [
            self.tag,
            idx,
            self.artifact_number,
            self.sanity_check,
            self._n1,
            self._N1,
            self.get_N1prop(),
            self._n2,
            self._N2,
            self.get_N2prop(),
            self.baseline,
            self._time_stamps[self._artifact_idx],
        ]
        for i in range(2):
            current_analysis.append(
                self.focus_x()[self.maxima_ids[i]] if len(self.maxima_ids) > i else None
            )
            current_analysis.append(
                self.focus_y()[self.maxima_ids[i]] if len(self.maxima_ids) > i else None
            )
            current_analysis.append(
                self.focus_x()[self.minima_ids[i]] if len(self.minima_ids) > i else None
            )
            current_analysis.append(
                self.focus_y()[self.minima_ids[i]] if len(self.minima_ids) > i else None
            )
        return current_analysis

    def _plt_n(self, show_label=False):
        if not self.sanity_check:
            return
        maxima_ids = self.maxima_ids
        minima_ids = self.minima_ids
        focus_x = self.focus_x()
        focus_y = self.sfocus_y()
        sfocus_y = self.sfocus_y()

        # XXX PERFORMING MEASUREMENTS ON SMOOTHENED FUNCTION XXX
        # focus_y = sfocus_y.copy()
        # XXX ---------------------------------------------- XXX

        if show_label:
            msl = 0.5
            lwl = 0.3
            msn = 0.5
            lwn = 0.3
            label1 = f"$n_1$ = {self._n1:.0f} $\mu$V, $N_1$ = {self._N1:.0f} nVs"
            label2 = f"$n_2$ = {self._n2:.0f} $\mu$V, $N_2$ = {self._N2:.0f} nVs"
            plt.annotate(
                "$n_1$",
                xy=(
                    focus_x[minima_ids[0]] + 0.05,
                    (self._c1y + focus_y[self.minima_ids[0]]) / 2,
                ),
            )
            plt.annotate(
                "$n_2$",
                xy=(
                    focus_x[minima_ids[1]] + 0.05,
                    (self._c2y + focus_y[self.minima_ids[1]]) / 2,
                ),
            )
        else:
            msl = 0
            lwl = 0.3
            msn = 2
            lwn = 1.0
            label1 = ""
            label2 = ""

        maxmax1x = [focus_x[maxima_ids[0]], self._t1x]
        maxmax1y = [focus_y[maxima_ids[0]], self._t1y]
        n1x = [focus_x[self.minima_ids[0]], focus_x[self.minima_ids[0]]]
        n1y = [focus_y[self.minima_ids[0]], self._c1y]
        plt.plot(maxmax1x, maxmax1y, "c.:", markersize=msl, linewidth=lwl)
        plt.plot(n1x, n1y, "c.-", markersize=msn, linewidth=lwn, label=label1)
        f1x = focus_x[self.maxima_ids[0] : self._tid1]
        f1y = focus_y[self.maxima_ids[0] : self._tid1]
        f1l = self._get_slope_point(
            focus_x[maxima_ids[0]],
            focus_y[maxima_ids[0]],
            self._t1x,
            self._t1y,
            focus_x[self.maxima_ids[0] : self._tid1],
        )
        plt.fill_between(f1x, f1y, f1l, color="c", alpha=0.2)
        maxmax2x = [focus_x[self.maxima_ids[1]], self._t2x]
        maxmax2y = [focus_y[self.maxima_ids[1]], self._t2y]
        n2x = [focus_x[self.minima_ids[1]], focus_x[self.minima_ids[1]]]
        n2y = [focus_y[self.minima_ids[1]], self._c2y]
        plt.plot(maxmax2x, maxmax2y, "c.:", markersize=msl, linewidth=lwl)
        plt.plot(n2x, n2y, "c.-", markersize=msn, linewidth=lwn, label=label2)
        f2x = focus_x[maxima_ids[1] : self._tid2]
        f2y = focus_y[maxima_ids[1] : self._tid2]
        f2l = self._get_slope_point(
            focus_x[self.maxima_ids[1]],
            focus_y[self.maxima_ids[1]],
            self._t2x,
            self._t2y,
            focus_x[self.maxima_ids[1] : self._tid2],
        )
        plt.fill_between(f2x, f2y, f2l, color="c", alpha=0.2)

    def plot_artifact(self):
        artifact_x = self._time_stamps[self._artifact_idx]
        maxima_ids = self.maxima_ids
        minima_ids = self.minima_ids
        focus_x = self.focus_x()
        focus_y = self.focus_y()
        sfocus_y = self.sfocus_y()

        fig = plt.figure(figsize=(10, 4))

        plt.axvline(
            x=artifact_x,
            color="y",
            linewidth=0.5,
            label=f"artifact position: {artifact_x:.2f} ms",
        )
        plt.axhline(
            y=self.baseline,
            color="y",
            linestyle=":",
            linewidth=0.5,
            label=f"baseline: {self.baseline:.0f} $\mu$V",
        )
        plt.axhline(
            y=0,
            color="k",
            linestyle=":",
            linewidth=0.5,
        )
        plt.plot(self._time_stamps, self._data, "b.", markersize=3, label="measurement")
        plt.plot(
            self._baseline_x,
            self._baseline_y,
            "y.",
            markersize=1,
        )
        plt.plot(self.focus_x(), self.focus_y(), "y.", markersize=1)
        plt.plot(
            self.focus_x(),
            self.sfocus_y(),
            "k-",
            linewidth=0.25,
            label=f"{self._smoothing_window}-point moving average",
        )
        self._plt_n(True)
        plt.axhline(
            y=self._artifact_threshold,
            color="m",
            linestyle="-",
            linewidth=1.0,
            label="artifact threshold",
        )
        plt.title(f"{self.series}/{self.tag}_{self.artifact_number}")
        plt.xlim([self._time_stamps[0], self._time_stamps[-1]])
        # plt.ylim([-3000, 800])
        plt.ylim([-1000, 200])
        plt.xlabel("Time [ms]")
        plt.ylabel("Amplitude [$\mu$V]")
        plt.legend(loc=4)
        fig.tight_layout()
        plt.savefig(
            f"{self._res_dir}{self.series}/{self.tag}_{self.artifact_number}{self._png_ext}",
            dpi=300,
        )
        plt.close()

    def __str__(self):
        return f"artifact_analysis class. Baseline={self.baseline}, {np.std(np.diff(self.focus_y()))}"
