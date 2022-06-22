import os


def get_file_name(path="", withExtension=False):
    """
    Get file name from directory path
    :param path: Directory path
    :param withExtension: if true, output returns with the file extension
    :return: filenamwe
    """

    if len(path) == 0:
        return ""

    pathname, extension = os.path.splitext(path)

    filename = pathname.split('/')[-1]

    if withExtension:
        return filename + extension
    else:
        return filename


