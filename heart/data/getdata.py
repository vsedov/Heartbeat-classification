import os
import zipfile

from heart.constants import DIR
from heart.log import get_logger

log = get_logger(__name__)

zip_path = f"{DIR}/data/zip/"
command = f"kaggle datasets download -d shayanfazeli/heartbeat -p {zip_path}"


def check_folder(folder):
    return True if os.path.isdir(folder) and os.listdir(folder) else False


def check_current_path_file(filename):
    return True if str(os.path.exists(filename)) else False


def check_zip():
    if check_current_path_file(f"{zip_path}heartbeat.zip"):
        log.info("Downloading zip file")
        os.system(command)


def force_replace_zip():
    if check_folder(zip_path):
        for files in os.listdir(zip_path):
            os.remove(files)
    check_zip()


def download_dir():
    if check_folder(f"{DIR}/data/heartbeat"):
        return True

    check_zip()

    with zipfile.ZipFile(f"{zip_path}heartbeat.zip", "r") as zip_ref:
        zip_ref.extractall(f"{DIR}/data/heartbeat")
