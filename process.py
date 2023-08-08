import os
import re
from collections import defaultdict
from glob import glob

import numpy as np
import pandas as pd
from numpy import genfromtxt
from tqdm import tqdm


def interpolate_numeric_df(df: pd.DataFrame, timestamp_col: str, new_timestamp: np.ndarray) -> pd.DataFrame:
    """
    Interpolate a DF linearly.
    Args:
        df: input DF
        timestamp_col: timestamp column name in the DF
        new_timestamp: array of evaluated timestamps

    Returns:
        an interpolated DF
    """
    cols_except_ts = df.columns.to_list()
    cols_except_ts.remove(timestamp_col)

    df_value = df[cols_except_ts].to_numpy()
    df_timestamp = df[timestamp_col].to_numpy()

    new_df = {timestamp_col: new_timestamp}
    for i, col in enumerate(cols_except_ts):
        new_df[col] = np.interp(x=new_timestamp, xp=df_timestamp, fp=df_value[:, i])

    new_df = pd.DataFrame(new_df)
    return new_df


def sliding_window(data: np.ndarray, window_size: int, step_size: int, get_last: bool = True) -> np.ndarray:
    """
    Sliding window along the first axis of the input array.
    Args:
        data:
        window_size:
        step_size:
        get_last:

    Returns:

    """
    num_windows = (len(data) - window_size) / step_size + 1
    if num_windows < 1:
        return np.empty([0, window_size, *data.shape[1:]], dtype=data.dtype)

    # if possible, run fast sliding window
    if window_size % step_size == 0:
        result = np.empty([int(num_windows), window_size, *data.shape[1:]], dtype=data.dtype)
        div = int(window_size / step_size)
        for window_idx, data_idx in enumerate(range(0, window_size, step_size)):
            new_window_data = data[data_idx:data_idx + (len(data) - data_idx) // window_size * window_size].reshape(
                [-1, window_size, *data.shape[1:]])

            new_window_idx = list(range(window_idx, int(num_windows), div))
            result[new_window_idx] = new_window_data
    # otherwise, run a regular loop
    else:
        result = np.array([data[i:i + window_size] for i in range(0, len(data) - window_size + 1, step_size)])

    if get_last and (num_windows % 1 != 0):
        result = np.concatenate([result, [data[-window_size:]]])

    return result


def shifting_window(data: np.ndarray, window_size: int, max_num_windows: int, min_step_size: int,
                    start_idx: int, end_idx: int) -> np.ndarray:
    """
    Get window(s) from an array while ensuring a certain part of the array is included.

    Args:
        data: array shape [data length, ...]
        window_size: window size
        max_num_windows: desired number of windows, the actual number returned maybe smaller,
            depending on `min_step_size`
        min_step_size: only get multiple windows if 2 windows are farther than `min_step_size` from each other
        start_idx: start index of the required part (inclusive)
        end_idx: end index of the required part (inclusive)

    Returns:
        array shape [num window, window size, ...]
    """
    end_idx = end_idx + 1

    # exception 1: data array is too small compared to window size
    if len(data) < window_size:
        return np.empty([0, window_size, *data.shape[1:]], dtype=data.dtype)
    elif len(data) == window_size:
        return np.expand_dims(data, axis=0)

    # exception 2: required part is not entirely in data array
    end_idx = min(end_idx, len(data))
    start_idx = max(start_idx, 0)

    # data array is large enough to contain both required part and window size
    first_window_start_idx = max(min(end_idx - window_size, start_idx), 0)
    last_window_start_idx = min(max(end_idx - window_size, start_idx), len(data) - window_size)
    assert last_window_start_idx >= first_window_start_idx

    # if there's only 1 window
    if first_window_start_idx == last_window_start_idx:
        windows = np.expand_dims(data[first_window_start_idx:first_window_start_idx + window_size], axis=0)
        return windows

    # otherwise, get windows by linspace
    max_num_windows = min(max_num_windows,
                          np.ceil((last_window_start_idx - first_window_start_idx + 1) / min_step_size).astype(int))
    window_start_indices = np.linspace(first_window_start_idx, last_window_start_idx, num=max_num_windows,
                                       endpoint=True, dtype=int)

    windows = np.array([data[i:i + window_size] for i in window_start_indices])
    return windows


class Process:
    def __init__(self, name: str, raw_folder: str, destination_folder: str,
                 signal_freq: float = 50., window_size_sec: float = 4):
        """
        This class transforms public datasets into the same format for ease of use.

        Args:
            name: name of the dataset
            raw_folder: path to unprocessed dataset
            destination_folder: folder to save output
            signal_freq: (Hz) resample signal to this frequency by linear interpolation
            window_size_sec: window size in second
        """
        self.name = name
        self.raw_folder = raw_folder
        self.destination_folder = destination_folder
        # pattern for output npy file name
        self.output_name_pattern = f'{destination_folder}/{name}_{{label}}/{{index}}.npy'

        # convert sec to num rows
        self.window_size_row = int(window_size_sec * signal_freq)
        # convert Hz to sample/msec
        self.signal_freq = signal_freq / 1000

    def resample(self, df: pd.DataFrame, timestamp_col: str) -> pd.DataFrame:
        """
        Resample a dataframe

        Args:
            df: dataframe
            timestamp_col: name of the timestamp column

        Returns:
            resampled dataframe
        """
        start_ts = df.at[0, timestamp_col]
        end_ts = df.at[len(df) - 1, timestamp_col]

        # get new timestamp array (unit: msec)
        new_ts = np.arange(np.floor((end_ts - start_ts) * self.signal_freq + 1)) / self.signal_freq + start_ts
        new_ts = new_ts.astype(int)

        # interpolate
        df = interpolate_numeric_df(df, timestamp_col=timestamp_col, new_timestamp=new_ts)

        return df

    def write_npy_sequences(self, data: list, label: str):
        """
        Write all sequences into npy files

        Args:
            data: list of np arrays, each array is a sequence of shape [num windows, window length,
            10(msec,feature_x,feature_y,feature_Z)]
            label: label of this data
        """
        num_digits = len(str(len(data)))
        index_pattern = f'%0{num_digits}d'

        for i, seq_windows in enumerate(data):
            output_path = self.output_name_pattern.format(label=label, index=index_pattern % i)
            os.makedirs(os.path.split(output_path)[0], exist_ok=True)
            print(f'writing {seq_windows.shape} to {output_path}')
            np.save(output_path, seq_windows)

    def run(self):
        """
        Main processing method
        """
        raise NotImplementedError()


class KFall(Process):
    FALL_TASK_ID = set('%02d' % n for n in range(20, 35))

    def __init__(self, max_window_per_fall: int = 3, min_fall_window_step: float = 0.5,
                 *args, **kwargs):
        """
        Args:
            max_window_per_fall: max number of windows to take for each fall event (as fall events are short)
            min_fall_window_step: (unit: second) minimum step size between fall windows
        """
        super().__init__(*args, **kwargs)
        self.max_window_per_fall = max_window_per_fall
        self.raw_data_folder = f'{self.raw_folder}/sensor_data'
        self.raw_label_folder = f'{self.raw_folder}/label_data'
        # minimum step size between fall windows
        self.min_fall_window_step_size = min(
            int(min_fall_window_step * self.signal_freq * 1000),
            self.window_size_row // 2
        )
        # make sure at least 1s of post-fall impact event is included in the window
        self.expand_after_impact = int(self.signal_freq * 1000)

    def _read_label(self, path: str) -> pd.DataFrame:
        """
        Read label file

        Args:
            path: path to file

        Returns:
            dataframe of labels
        """
        df = pd.read_excel(path)
        df['Task Code (Task ID)'] = df['Task Code (Task ID)'].map(
            lambda s: int(re.match(r'(?:\s+)?F(?:[0-9]+) \(([0-9]+)\)(?:\s+)?', s).group(1))
            if pd.notna(s) else None
        )
        df = df.fillna(method='ffill')
        df = df.rename({'Task Code (Task ID)': 'Task ID'}, axis=1)
        return df

    def _read_data(self, path: str) -> pd.DataFrame:
        """
        Read data file without interpolation because it must be unchanged for label matching

        Args:
            path: path to data csv file

        Returns:
            dataframe of sensor data
        """
        df = pd.read_csv(path, usecols=['TimeStamp(s)', 'FrameCounter', 'AccX', 'AccY', 'AccZ', 'GyrX', 'GyrY', 'GyrZ',
                                        'EulerX', 'EulerY', 'EulerZ'])
        df['TimeStamp(s)'] *= 1000
        df = df.rename({'TimeStamp(s)': 'msec'}, axis=1)
        return df

    def _get_session_info(self, session_id: str) -> tuple:
        """
        Get subject ID, task ID, trial ID from session ID

        Args:
            session_id: session ID (name of data file)

        Returns:
            (subject ID, task ID, trial ID), all are strings
        """
        res = re.match(r'S([0-9]+)T([0-9]+)R([0-9]+)', session_id)
        print(res)
        subject_id, task_id, trial_id = [res.group(i) for i in range(1, 4)]
        return subject_id, task_id, trial_id

    def _get_fall_window(self, data_df: pd.DataFrame, label_row: pd.Series) -> np.ndarray:
        """
        Turn a fall session DF into a numpy array of fall windows

        Args:
            data_df: data df
            label_row: a row in the label df, label of this data df

        Returns:
            numpy array shape [num windows, window length, 10(msec,feature_x,feature_y,feature_Z)]
        """
        # get label in msec
        fall_onset_frame = label_row.at['Fall_onset_frame']
        fall_impact_frame = label_row.at['Fall_impact_frame']
        frame_counter = data_df['FrameCounter'].to_numpy()
        fall_onset_msec = data_df.loc[(frame_counter == fall_onset_frame), 'msec'].iat[0]
        fall_impact_msec = data_df.loc[(frame_counter == fall_impact_frame), 'msec'].iat[0]
        assert fall_impact_msec > fall_onset_msec, 'fall_impact_msec must be > fall_onset_msec'

        # resample (indices change after this)
        data_df = self.resample(
            data_df[['msec', 'AccX', 'AccY', 'AccZ', 'GyrX', 'GyrY', 'GyrZ', 'EulerX', 'EulerY', 'EulerZ']],
            timestamp_col='msec')
        data_arr = data_df.to_numpy()

        # padding if not enough rows
        if len(data_arr) <= self.window_size_row:
            window = np.pad(data_arr, [[self.window_size_row - len(data_arr), 0], [0, 0]])
            return np.expand_dims(window, axis=0)

        # find start & end indices by msec
        fall_indices = np.nonzero((data_arr[:, 0] <= fall_impact_msec) & (data_arr[:, 0] >= fall_onset_msec))[0]
        fall_onset_idx = fall_indices[0]
        fall_impact_idx = fall_indices[-1]

        windows = shifting_window(
            data_arr,
            window_size=self.window_size_row, max_num_windows=self.max_window_per_fall,
            min_step_size=self.min_fall_window_step_size,
            start_idx=fall_onset_idx, end_idx=fall_impact_idx + self.expand_after_impact
        )
        return windows

    def _get_adl_windows(self, data_df: pd.DataFrame) -> np.ndarray:
        """

        Args:
            data_df:

        Returns:

        """
        data_df = self.resample(
            data_df[['msec', 'AccX', 'AccY', 'AccZ', 'GyrX', 'GyrY', 'GyrZ', 'EulerX', 'EulerY', 'EulerZ']],
            timestamp_col='msec')
        data_arr = data_df.to_numpy()
        if len(data_arr) >= self.window_size_row:
            windows = sliding_window(data_arr, window_size=self.window_size_row, step_size=self.window_size_row // 2)
        else:
            window = np.pad(data_arr, [[self.window_size_row - len(data_arr), 0], [0, 0]])
            windows = np.array([window])
        return windows

    def run(self):

        all_fall_sequences = []
        all_adl_sequences = defaultdict(list)

        # go through all subjects
        for subject_id in tqdm(sorted(os.listdir(self.raw_data_folder))):
            label_df = self._read_label(f'{self.raw_label_folder}/{subject_id}_label.xlsx')

            # for each session of this subject
            for session_file in sorted(glob(f'{self.raw_data_folder}/{subject_id}/*.csv')):
                session_file = os.path.normpath(session_file).replace('\\', '/')
                session_data_df = self._read_data(session_file)

                subject_id, task_id, trial_id = self._get_session_info(session_file.split('/')[-1][:-4])

                # if this is a fall session
                if task_id in self.FALL_TASK_ID:
                    label_row = label_df.loc[(label_df['Task ID'] == int(task_id)) &
                                             (label_df['Trial ID'] == int(trial_id))]
                    assert label_row.shape[0] == 1, f'Each session is represented by only 1 row in the label file. ' \
                                                    f'But {label_row.shape[0]} rows found for session {session_file}'
                    fall_window = self._get_fall_window(session_data_df, label_row.iloc[0])
                    all_fall_sequences.append(fall_window)
                # if this is an ADL session
                else:
                    adl_windows = self._get_adl_windows(session_data_df)
                    all_adl_sequences[f'task{task_id}'].append(adl_windows)

        # write fall data
        self.write_npy_sequences(all_fall_sequences, label='fall')
        # write adl data
        for task_id, task_data in all_adl_sequences.items():
            self.write_npy_sequences(task_data, label=task_id)


class FallAllD(Process):
    FALL_TASK_ID = set('%02d' % n for n in range(101, 136))

    def _read_data(self, add):
        # Change the working directory to the specified 'add'
        os.chdir(add)

        # Get a list of all file names in the 'add' directory
        FileNamesAll = os.listdir(add)

        # Filter out file names that end with '_A.dat'
        FileNames = []
        for f_name in FileNamesAll:
            if f_name.endswith('_A.dat'):
                FileNames.append(f_name)
        LL = len(FileNames)

        # Initialize lists to store various data attributes
        l_SubjectID = []
        l_Device = []
        l_ActivityID = []
        l_TrialNo = []
        l_Acc = []
        l_Gyr = []
        l_Mag = []
        l_Bar = []

        # Loop through each file and extract relevant information
        for i in range(LL):
            f_name = FileNames[i]
            # Extract SubjectID from the file name
            SubjectID = int(f_name[1:3])
            l_SubjectID.append(np.uint8(SubjectID))

            # Extract ActivityID from the file name
            ActivityID = int(f_name[8:11])
            l_ActivityID.append(np.uint8(ActivityID))

            # Extract TrialNo from the file name
            TrialNo = int(f_name[13:15])
            l_TrialNo.append(np.uint8(TrialNo))

            # Determine the device type based on the file name
            Device = ''
            if (int(f_name[5]) == 1):
                Device = 'Neck'
            else:
                if (int(f_name[5]) == 2):
                    Device = 'Wrist'
                else:
                    Device = 'Waist'
            l_Device.append(Device)

            # Load accelerometer data from the file
            l_Acc.append(np.int16(genfromtxt(f_name, delimiter=',')))
            chArr = list(f_name)
            chArr[16] = 'G'
            f_name = "".join(chArr)
            l_Gyr.append(np.int16(genfromtxt(f_name, delimiter=',')))

            # Load gyroscope data from the file
            chArr = list(f_name)
            chArr[16] = 'M'
            f_name = "".join(chArr)
            l_Mag.append(np.int16(genfromtxt(f_name, delimiter=',')))

            # Load magnetometer data from the file
            chArr = list(f_name)
            chArr[16] = 'B'
            f_name = "".join(chArr)
            l_Bar.append(genfromtxt(f_name, delimiter=','))
            print(f'File  {i + 1}  out of {len(FileNames)}')

        activity_id_series = pd.Series(l_ActivityID, dtype=np.int16)
        FallAllD = pd.concat([
            pd.Series(l_SubjectID),
            pd.Series(l_Device),
            activity_id_series,
            pd.Series(l_TrialNo),
            pd.Series(l_Acc),
            pd.Series(l_Gyr),
            pd.Series(l_Mag),
            pd.Series(l_Bar),
        ], axis=1)

        FallAllD.columns = ['SubjectID', 'Device', 'ActivityID', 'TrialNo', 'Acc', 'Gyr', 'Mag', 'Bar']

        return FallAllD

    def process_device_data(self, device, msec1, msec2, index, fall_all_d, crop_time=None):
        """
        Process the device data and return a DataFrame containing the processed data.

        Args:
            device (str): The device type ('Neck', 'Waist', or 'Wrist').
            msec1 (numpy.array): The array containing the msec values for df1.
            msec2 (numpy.array): The array containing the msec values for df2.
            index (int): The index of the selected row in the DataFrame.
            fall_all_d (DataFrame): The DataFrame containing the data.

        Returns:
            DataFrame: A DataFrame containing the processed device data.
            [msec,Neck[9], Wrist[9], Waist[9]]
        """
        if device == 'Neck':
            neck_Acc_x = fall_all_d['Acc'][index][:, 0]
            neck_Acc_y = fall_all_d['Acc'][index][:, 1]
            neck_Acc_z = fall_all_d['Acc'][index][:, 2]

            neck_Gyr_x = fall_all_d['Gyr'][index][:, 0]
            neck_Gyr_y = fall_all_d['Gyr'][index][:, 1]
            neck_Gyr_z = fall_all_d['Gyr'][index][:, 2]

            df1 = pd.DataFrame({
                'msec': msec1,
                'neck_Acc_x': neck_Acc_x,
                'neck_Acc_y': neck_Acc_y,
                'neck_Acc_z': neck_Acc_z,
                'neck_Gyr_x': neck_Gyr_x,
                'neck_Gyr_y': neck_Gyr_y,
                'neck_Gyr_z': neck_Gyr_z,
            })

            neck_Mag_x = fall_all_d['Mag'][index][:, 0]
            neck_Mag_y = fall_all_d['Mag'][index][:, 1]
            neck_Mag_z = fall_all_d['Mag'][index][:, 2]

            df2 = pd.DataFrame({
                'msec': msec2,
                'neck_Mag_x': neck_Mag_x,
                'neck_Mag_y': neck_Mag_y,
                'neck_Mag_z': neck_Mag_z,
            })

        elif device == 'Waist':
            waist_Acc_x = fall_all_d['Acc'][index][:, 0]
            waist_Acc_y = fall_all_d['Acc'][index][:, 1]
            waist_Acc_z = fall_all_d['Acc'][index][:, 2]

            waist_Gyr_x = fall_all_d['Gyr'][index][:, 0]
            waist_Gyr_y = fall_all_d['Gyr'][index][:, 1]
            waist_Gyr_z = fall_all_d['Gyr'][index][:, 2]

            df1 = pd.DataFrame({
                'msec': msec1,
                'waist_Acc_x': waist_Acc_x,
                'waist_Acc_y': waist_Acc_y,
                'waist_Acc_z': waist_Acc_z,
                'waist_Gyr_x': waist_Gyr_x,
                'waist_Gyr_y': waist_Gyr_y,
                'waist_Gyr_z': waist_Gyr_z,
            })

            waist_Mag_x = fall_all_d['Mag'][index][:, 0]
            waist_Mag_y = fall_all_d['Mag'][index][:, 1]
            waist_Mag_z = fall_all_d['Mag'][index][:, 2]

            df2 = pd.DataFrame({
                'msec': msec2,
                'waist_Mag_x': waist_Mag_x,
                'waist_Mag_y': waist_Mag_y,
                'waist_Mag_z': waist_Mag_z,
            })

        elif device == 'Wrist':
            wrist_Acc_x = fall_all_d['Acc'][index][:, 0]
            wrist_Acc_y = fall_all_d['Acc'][index][:, 1]
            wrist_Acc_z = fall_all_d['Acc'][index][:, 2]

            wrist_Gyr_x = fall_all_d['Gyr'][index][:, 0]
            wrist_Gyr_y = fall_all_d['Gyr'][index][:, 1]
            wrist_Gyr_z = fall_all_d['Gyr'][index][:, 2]

            df1 = pd.DataFrame({
                'msec': msec1,
                'wrist_Acc_x': wrist_Acc_x,
                'wrist_Acc_y': wrist_Acc_y,
                'wrist_Acc_z': wrist_Acc_z,
                'wrist_Gyr_x': wrist_Gyr_x,
                'wrist_Gyr_y': wrist_Gyr_y,
                'wrist_Gyr_z': wrist_Gyr_z,
            })

            wrist_Mag_x = fall_all_d['Mag'][index][:, 0]
            wrist_Mag_y = fall_all_d['Mag'][index][:, 1]
            wrist_Mag_z = fall_all_d['Mag'][index][:, 2]

            df2 = pd.DataFrame({
                'msec': msec2,
                'wrist_Mag_x': wrist_Mag_x,
                'wrist_Mag_y': wrist_Mag_y,
                'wrist_Mag_z': wrist_Mag_z,
            })

        df1_resampled = self.resample(df1, timestamp_col='msec')
        df2_resampled = self.resample(df2, timestamp_col='msec')

        # Assumes both dataframes have the same 'msec' values after resampling
        merged_df = pd.merge(df1_resampled, df2_resampled, on='msec')
        if crop_time is not None:
            merged_df = merged_df[merged_df['msec'] <= 7000]
        return merged_df

    def run(self):
        FallAllD = self._read_data(self.raw_folder)

        # Calculate values for msec1 and msec2
        msec1 = np.arange(0, 20, 1 / 238) * 1000
        msec2 = np.arange(0, 20, 1 / 80) * 1000

        # Initialize lists to store sequences
        all_fall_sequences = []
        all_adl_sequences = defaultdict(list)

        # Get unique subject, activity, and trial IDs
        unique_subjects = FallAllD['SubjectID'].drop_duplicates().tolist()
        unique_activities = FallAllD['ActivityID'].drop_duplicates().tolist()
        unique_trials = FallAllD['TrialNo'].drop_duplicates().tolist()

        # Get the last 35 unique ActivityIDs
        last_35_activities = sorted(unique_activities)[-35:]

        # Loop through unique subjects, activities, and trials
        for sub in unique_subjects:
            for act in unique_activities:
                for tri in unique_trials:
                    # Get the subset of data based on specific conditions
                    subset_df = FallAllD.loc[(FallAllD['SubjectID'] == sub) &
                                             (FallAllD['ActivityID'] == act) &
                                             (FallAllD['TrialNo'] == tri)]

                    label = 'non_fall'
                    # Check if subset_df is not empty and 'Device' has 3 unique values
                    if not subset_df.empty and subset_df['Device'].nunique() == 3:
                        merged = None
                        dataframes_to_merge = []

                        # Loop through the subset_df and process each device data
                        for index, value in subset_df.iterrows():
                            if act in last_35_activities:
                                # Limit data to 8000ms for specific ActivityIDs
                                temp_df = self.process_device_data(value['Device'], msec1, msec2, index, subset_df,
                                                                   8000)
                                label = 'fall'
                            elif act in [102, 104, 108, 110, 116, 118, 120, 122, 124, 126, 128]:
                                # Limit data to 7000ms for specific ActivityIDs
                                temp_df = self.process_device_data(value['Device'], msec1, msec2, index, subset_df,
                                                                   8000)
                            else:
                                temp_df = self.process_device_data(value['Device'], msec1, msec2, index, subset_df)

                            dataframes_to_merge.append(temp_df)

                        # Merge the processed dataframes
                        if dataframes_to_merge:
                            merged = pd.merge(dataframes_to_merge[0], dataframes_to_merge[1], on='msec')
                            merged = pd.merge(merged, dataframes_to_merge[2], on='msec')
                            merged_arr = merged.to_numpy()

                            # Append sequences based on the label
                            if label == 'fall':
                                windows = sliding_window(merged_arr, window_size=self.window_size_row,
                                                         step_size=int(0.5 * self.signal_freq * 1000))
                                all_fall_sequences.append(windows)
                            elif label == 'non_fall':
                                windows = sliding_window(merged_arr, window_size=self.window_size_row,
                                                         step_size=self.window_size_row // 2)
                                all_adl_sequences[f'act{act}'].append(windows)

        # write fall data
        self.write_npy_sequences(all_fall_sequences, label='fall')
        # write adl data
        for act_id, act_data in all_adl_sequences.items():
            self.write_npy_sequences(act_data, label=act_id)


class Erciyes(Process):
    def process_txt_file(self, file_path, prefix):
        """
        Process a TXT file containing sensor data.

        Args:
            file_path (str): Path to the TXT file.
            prefix (str): Prefix to add to column names.

        Returns:
            pd.DataFrame: Processed DataFrame with renamed columns and resampled timestamps.
        """
        # Columns to read from the TXT file
        cols_to_read = ['Counter', 'Acc_X', 'Acc_Y', 'Acc_Z', 'Gyr_X', 'Gyr_Y', 'Gyr_Z', 'Mag_X', 'Mag_Y', 'Mag_Z',
                        'Roll', 'Pitch', 'Yaw']

        # Read the TXT file into a DataFrame, skip the first 4 rows, and select specified columns
        df = pd.read_csv(file_path, delimiter='\t', skiprows=4, usecols=cols_to_read)

        # Convert the 'Counter' values to milliseconds (25Hz frequency)
        df['Counter'] = df['Counter'] * 40

        # Rename columns and add the provided prefix to each column name
        new_columns = {
            'Counter': 'msec',
            'Acc_X': prefix + 'Acc_x',
            'Acc_Y': prefix + 'Acc_y',
            'Acc_Z': prefix + 'Acc_z',
            'Gyr_X': prefix + 'Gyr_x',
            'Gyr_Y': prefix + 'Gyr_y',
            'Gyr_Z': prefix + 'Gyr_z',
            'Mag_X': prefix + 'Mag_X',
            'Mag_Y': prefix + 'Mag_Y',
            'Mag_Z': prefix + 'Mag_Z',
            'Roll': prefix + 'euler_z',
            'Pitch': prefix + 'euler_x',
            'Yaw': prefix + 'euler_y'
        }
        df.rename(columns=new_columns, inplace=True)

        # Rename columns and add the provided prefix to each column name
        df = self.resample(df, timestamp_col='msec')
        return df

    def run(self):
        for folder in os.listdir(self.raw_folder):
            gender = 'man' if folder.startswith('1') else 'woman' if folder.startswith('2') else None
            if gender:
                test_export_path = os.path.join(self.raw_folder, folder, 'Testler Export')
                for sub_folder in os.listdir(test_export_path):
                    test_path = os.path.join(test_export_path, sub_folder)
                    for test in os.listdir(test_path):
                        if 'Fail' in test:
                            continue
                        activity = 'fall' if sub_folder.startswith('9') else 'non_fall' if sub_folder.startswith(
                            '8') else None
                        if activity:
                            waist_file_path = os.path.join(test_path, test, '340535.txt')
                            wrist_file_path = os.path.join(test_path, test, '340537.txt')
                            print(wrist_file_path)
                            if os.path.exists(waist_file_path) and os.path.exists(wrist_file_path):
                                waist_df = self.process_txt_file(waist_file_path, 'waist_')
                                wrist_df = self.process_txt_file(wrist_file_path, 'wrist_')

                                combined_df = pd.merge(waist_df, wrist_df, on='msec')

                                output_dir = os.path.join(self.destination_folder, gender, activity)
                                os.makedirs(output_dir, exist_ok=True)

                                # Generate the next file number for this directory
                                existing_files = os.listdir(output_dir)
                                existing_numbers = [int(f.split('.')[0]) for f in existing_files if
                                                    f.split('.')[0].isdigit()]
                                next_number = max(existing_numbers, default=0) + 1

                                # combined_df.to_csv(os.path.join(output_dir, sub_folder + '.csv'), index=False)
                                output_path = os.path.join(output_dir, str(next_number) + '.parquet')
                                combined_df.to_parquet(output_path, index=False)


if __name__ == '__main__':
    # for sec in [2.5, 4, 5, 6, 7, 8]:
    #     KFall(
    #         max_window_per_fall=3,
    #         raw_folder='C:/Repository/master/Raw_Dataset/KFall',
    #         name='KFall',
    #         destination_folder=f'C:/Repository/master/Processed_Dataset/KFall/KFall_window_sec{sec}',
    #         signal_freq=50, window_size_sec=sec
    #     ).run()

    for sec in range(4, 9):
        FallAllD(
            raw_folder='C:/Repository/master/Raw_Dataset/FallAllD/FallAllD__zip/FallAllD',
            name='FallAllD',
            destination_folder=f'C:/Repository/master/Processed_Dataset/FallAllD/FallAllD_window_sec{sec}',
            signal_freq=50, window_size_sec=sec
        ).run()

    # Erciyes(
    #     raw_folder=
    #     'C:/Repository/master/Raw_Dataset/Erciyes/simulated+falls+and+daily+living+activities+data+set/Tests',
    #     name='Erciyes',
    #     destination_folder=f'C:/Repository/master/Processed_Dataset/Erciyes',
    #     signal_freq=50
    # ).run()
