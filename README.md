# sscm_reader

Converts sound sensor `*.sscm` files to pandas DataFrames.

## Installation

```bash
pip install sscm_reader
```

## Usage

```python
from sscm_reader import read_sscm, read_sscm_folder

sensor_name, loudness, sharpness, source, voltage, event_log = read_sscm(
    "path/to/file.sscm",
    add_tz_hours=0,
)

sensor_names, merged_loudness, merged_sharpness, merged_sources, merged_voltage, merged_event_log = read_sscm_folder(
    "path/to/folder",
    add_tz_hours=0,
)
```
