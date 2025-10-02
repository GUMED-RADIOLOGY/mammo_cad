# print_dicom_header.py

from pydicom import dcmread
from pydicom.errors import InvalidDicomError
from pathlib import Path
import sys

# Input DICOM file (change if needed)
DICOM_FILE = Path(r"/data/UC6/ECI_GUM_S0055/exp_ECI_GUM_S0055_20160525/scans/2-SINGLE_IMAGES__Mammografia_Diagnostyka__Diagnosis/resources/annotations/files/event_4d4ea55c-6da8-4d59-a3b3-8c3bcefbc092/segmentation.dcm")

def main():
    if not DICOM_FILE.exists():
        sys.stderr.write(f"Error: file not found: {DICOM_FILE}\n")
        sys.exit(1)

    try:
        ds = dcmread(DICOM_FILE, force=False)  # set force=True if needed
    except InvalidDicomError as e:
        sys.stderr.write(f"Error: not a valid DICOM file: {DICOM_FILE}\nDetails: {e}\n")
        sys.exit(2)

    # Save header to a text file in the current folder
    output_file = Path.cwd() / "dicom_header.txt"
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(f"--- DICOM Header: {DICOM_FILE.name} ---\n")
        f.write(str(ds))  # writes the dataset header only

    print(f"DICOM header saved to {output_file}")

if __name__ == "__main__":
    main()
