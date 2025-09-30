# print_dicom_header.py
# Requires: pip install pydicom

from pydicom import dcmread
from pydicom.errors import InvalidDicomError
from pathlib import Path
import sys

# >>> Put your DICOM file path here <<<
DICOM_FILE = Path(r"/data/UC6/ECI_GUM_S0001/exp_ECI_GUM_S0001_20190122/scans/2-SINGLE_IMAGES__Mammografia_Diagnostyka__Diagnosis/resources/DICOM/files/1.2.276.0.7230010.3.1.4.3059715276.5112.1678755402.13002.dcm")

def main():
    if not DICOM_FILE.exists():
        sys.stderr.write(f"Error: file not found: {DICOM_FILE}\n")
        sys.exit(1)

    try:
        ds = dcmread(DICOM_FILE, force=False)  # set force=True if your file lacks preamble
    except InvalidDicomError as e:
        sys.stderr.write(f"Error: not a valid DICOM file: {DICOM_FILE}\nDetails: {e}\n")
        sys.exit(2)

    # Pretty print the DICOM header (data elements)
    print(f"--- DICOM Header: {DICOM_FILE} ---")
    print(ds)
    print("\n--- Full elements ---")

if __name__ == "__main__":
    main()
