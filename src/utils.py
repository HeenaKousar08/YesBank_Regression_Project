import os


def create_folders():

    folders = [
        'models',
        'outputs',
        'outputs/graphs',
        'outputs/reports',
        'app'
    ]

    for folder in folders:

        if not os.path.exists(folder):
            os.makedirs(folder)