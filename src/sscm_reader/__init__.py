import pandas as pd
import struct
import os
import glob

labels = ["vehicle", "honking", "aircraft", "siren", "human",
          "bark", "bird", "church", "music", "wind", "rain"]


def read_header(f):
    """
    Reads and validates the header section of an SSCM file.

    Takes a pointer to an SSCM file and reads its header.
    Checks if the file format is correct and if the format version is supported.
    If the file seems valid, various metadata are returned.

    Args:
        f: Pointer to file already opened in binary mode

    Returns:
        tuple: (format_version, file_created_ts, sensor_name, num_source_classes)
            - format_version (str): Version of the SSCM file format
            - file_created_ts (int): Unix timestamp when file was created
            - sensor_name (str): Name of the sensor that created the file
            - num_source_classes (int): Number of sound source classification classes

    Raises:
        RuntimeError: If magic bytes don't match expected format
        RuntimeError: If file format version is not supported
        RuntimeError: If number of source classes doesn't match expected number of labels
    """
    magic = f.read(20)
    if magic != b'\x00\x00cityai_sc_sensor_v':
        raise RuntimeError(f'Unexpected file format: {magic}')

    format_version = f.read(2).decode("ascii")
    if (format_version != '01'):
        raise RuntimeError(
            'SSCM format version not supported: ' + format_version)

    file_created_ts = struct.unpack('I', f.read(4))[0]

    sensor_name_len = struct.unpack('B', f.read(1))[0]
    sensor_name = f.read(sensor_name_len).decode("ascii")[:sensor_name_len]

    num_source_classes = struct.unpack('B', f.read(1))[0]
    if (num_source_classes != len(labels)):
        raise RuntimeError(
            f'Specified labels ({len(labels)}) do not match number of labels in file header ({num_source_classes})!')

    return format_version, file_created_ts, sensor_name, num_source_classes


def read_sscm(path, add_tz_hours=None):
    """
    Reads one SSCM file and returns its content as DataFrames.

    Parses a single SSCM file, extracting all sensor data including loudness measurements,
    sharpness values, sound source classifications, voltage readings, and event logs.
    Converts timestamps to pandas datetime objects and optionally applies timezone offset.

    Args:
        path (str): Path to the SSCM file to read
        add_tz_hours (int, optional): Hours to add for timezone adjustment

    Returns:
        tuple: (sensor_name, loudness, sharpness, source, voltage, event_log)
            - sensor_name (str): Name of the sensor that created the file
            - loudness (pd.DataFrame): Loudness measurements with columns ['time', 'dba', 'spl_a']
            - sharpness (pd.DataFrame): Sharpness values with columns ['time', 'sharpness']
            - source (pd.DataFrame): Sound source classifications with time and probability columns
            - voltage (pd.DataFrame): Voltage readings with columns ['time', 'mV']
            - event_log (pd.DataFrame): System events with columns ['time', 'event', 'event value']

    Raises:
        RuntimeError: If file format is invalid or unsupported entry type is encountered
    """
    loudness = []
    sharpness = []
    source = []
    voltage = []
    event_log = []

    f = open(path, mode="rb")
    f.seek(0, 2)
    file_size = f.tell()
    f.seek(0)
    format_version, file_created_ts, sensor_name, num_source_classes = read_header(
        f)
    while f.tell() < file_size:
        block = f.read(2)
        if (block != b'\xff\xff'):
            f.seek(f.tell()-2)
        entry_type = struct.unpack('B', f.read(1))[0]
        time_ms = struct.unpack('q', f.read(8))[0]
        if entry_type == 0:
            dba = struct.unpack('f', f.read(4))[0]
            _ = struct.unpack('f', f.read(4))[0]
            # convert to linear SPL
            try:
                spl_a = 10 ** (dba / 10)
                loudness.append([time_ms, dba, spl_a])
            except Exception as e:
                print(f"Error converting dBA to SPL after {len(loudness)} entries")
                print(f"Entry type: {entry_type}, Time: {time_ms} ({pd.to_datetime(time_ms, unit='ms')}), dBA: {dba}")
        elif entry_type == 1:
            s = [time_ms]
            for _ in range(num_source_classes):
                s.append(struct.unpack('f', f.read(4))[0])
            source.append(s)
        elif entry_type == 2:
            sharp = struct.unpack('f', f.read(4))[0]
            sharpness.append([time_ms, sharp])
        elif entry_type == 100:
            mV = struct.unpack('H', f.read(2))[0]
            voltage.append([time_ms, mV])
        elif entry_type == 110:
            event_log.append([time_ms, "TIME_FROM_NTP", 0])
            pass
        elif entry_type == 111:
            event_log.append([time_ms, "TIME_FROM_RTC", 0])
            pass
        elif entry_type == 120:
            seconds = struct.unpack('H', f.read(2))[0]
            event_log.append([time_ms, "ENTER_SLEEP", seconds])
            pass
        elif entry_type == 121:
            sampling_rate = struct.unpack('f', f.read(4))[0]
            event_log.append([time_ms, "NIGHTLY_PD", sampling_rate])
            pass
        else:
            print(f"Path: {path}, Entry type: {entry_type}, Time: {time_ms} ({pd.to_datetime(time_ms, unit='ms')})")
            raise RuntimeError('Invalid entry type: ' + str(entry_type))

    loudness = pd.DataFrame(loudness, columns=['time', 'dba', 'spl_a'])
    loudness['time'] = pd.to_datetime(loudness['time'], unit='ms')
    sharpness = pd.DataFrame(sharpness, columns=['time', 'sharpness'])
    sharpness['time'] = pd.to_datetime(sharpness['time'], unit='ms')
    source_columns = ['time']
    source_columns.extend(labels)
    source = pd.DataFrame(source, columns=source_columns)
    source['time'] = pd.to_datetime(source['time'], unit='ms')
    source['label'] = source.iloc[:, 1:].idxmax(axis=1)
    voltage = pd.DataFrame(voltage, columns=['time', 'mV'])
    voltage['time'] = pd.to_datetime(voltage['time'], unit='ms')
    event_log = pd.DataFrame(event_log, columns=['time', 'event', 'value'])
    event_log['time'] = pd.to_datetime(event_log['time'], unit='ms')

    if add_tz_hours:
        loudness['time'] = loudness['time'] + pd.DateOffset(hours=add_tz_hours)
        sharpness['time'] = sharpness['time'] + \
            pd.DateOffset(hours=add_tz_hours)
        source['time'] = source['time'] + pd.DateOffset(hours=add_tz_hours)
        voltage['time'] = voltage['time'] + pd.DateOffset(hours=add_tz_hours)
        event_log['time'] = event_log['time'] + \
            pd.DateOffset(hours=add_tz_hours)

    return sensor_name, loudness, sharpness, source, voltage, event_log


def read_sscm_folder(folder_path, add_tz_hours=None):
    """
    Reads all SSCM files from a folder and merges them into one dataframe.
    Can be used to read all SSCM files of one sensor at once.

    Parameters:
    folder_path (str): Path to the folder containing SSCM files
    add_tz_hours (int, optional): Hours to add for timezone adjustment

    Returns:
        tuple: (sensor_names, merged_loudness, merged_sharpness, merged_sources, merged_voltage, merged_event_log)
    """
    # Find all .sscm files in the folder
    sscm_files = glob.glob(os.path.join(folder_path, "*.sscm"))

    if not sscm_files:
        print(f"No SSCM files found in folder: {folder_path}")
        return None, None, None, None, None, None

    print(f"Found {len(sscm_files)} SSCM files to process")

    # Initialize lists to store merged data
    all_loudness = []
    all_sharpness = []
    all_sources = []
    all_voltage = []
    all_event_log = []
    sensor_names = []

    for file_path in sorted(sscm_files):
        try:
            print(f"Processing file: {os.path.basename(file_path)}")
            sensor_name, loudness, sharpness, sources, voltage, event_log = read_sscm(
                file_path, add_tz_hours)

            # Add filename and sensor info to each dataframe
            loudness['filename'] = os.path.basename(file_path)
            loudness['sensor_id'] = sensor_name
            sharpness['filename'] = os.path.basename(file_path)
            sharpness['sensor_id'] = sensor_name
            sources['filename'] = os.path.basename(file_path)
            sources['sensor_id'] = sensor_name
            voltage['filename'] = os.path.basename(file_path)
            voltage['sensor_id'] = sensor_name
            event_log['filename'] = os.path.basename(file_path)
            event_log['sensor_id'] = sensor_name

            # Append to lists
            all_loudness.append(loudness)
            all_sharpness.append(sharpness)
            all_sources.append(sources)
            all_voltage.append(voltage)
            all_event_log.append(event_log)
            sensor_names.append(sensor_name)

        except Exception as e:
            print(f"Error processing file {file_path}: {str(e)}")
            continue

    # Merge all dataframes
    try:
        merged_loudness = pd.concat(
            all_loudness, ignore_index=True) if all_loudness else pd.DataFrame()
        merged_sharpness = pd.concat(
            all_sharpness, ignore_index=True) if all_sharpness else pd.DataFrame()
        merged_sources = pd.concat(
            all_sources, ignore_index=True) if all_sources else pd.DataFrame()
        merged_voltage = pd.concat(
            all_voltage, ignore_index=True) if all_voltage else pd.DataFrame()
        merged_event_log = pd.concat(
            all_event_log, ignore_index=True) if all_event_log else pd.DataFrame()

        # Sort by time
        if not merged_loudness.empty:
            merged_loudness = merged_loudness.sort_values(
                'time').reset_index(drop=True)
        if not merged_sharpness.empty:
            merged_sharpness = merged_sharpness.sort_values(
                'time').reset_index(drop=True)
        if not merged_sources.empty:
            merged_sources = merged_sources.sort_values(
                'time').reset_index(drop=True)
        if not merged_voltage.empty:
            merged_voltage = merged_voltage.sort_values(
                'time').reset_index(drop=True)
        if not merged_event_log.empty:
            merged_event_log = merged_event_log.sort_values(
                'time').reset_index(drop=True)

        print(f"Successfully merged data from {len(sensor_names)} files")
        print(f"Total loudness records: {len(merged_loudness)}")
        print(f"Total sharpness records: {len(merged_sharpness)}")
        print(f"Total source records: {len(merged_sources)}")
        print(f"Total voltage records: {len(merged_voltage)}")
        print(f"Total event log records: {len(merged_event_log)}")

        return sensor_names, merged_loudness, merged_sharpness, merged_sources, merged_voltage, merged_event_log

    except Exception as e:
        print(f"Error merging dataframes: {str(e)}")
        return None, None, None, None, None, None
