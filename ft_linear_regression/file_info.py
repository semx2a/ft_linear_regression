import os


def get_file_info(path: str) -> dict:
    """
    get_file_info takes a file path and returns a dictionary with the file's
    information.

    Parameters:
    path (str): path of the file.

    Return Value:
    file_info (dict): a dictionary containing the file's information.
    """
    try:
        # Get file info
        if os.path.exists(path) is False:
            raise FileNotFoundError(f'Could not find file {path}')

        file_info = {
            "name": os.path.basename(path),
            "size": os.path.getsize(path),
            "path": os.path.abspath(path),
            "last_modified": os.path.getmtime(path),
            "created": os.path.getctime(path)
        }

        return file_info

    except FileNotFoundError as fnf:
        print(fnf)
        raise
    except Exception as e:
        print(f'Exception: {e}')
        raise
