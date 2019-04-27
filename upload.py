import os, sys
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
from google.colab import auth
from oauth2client.client import GoogleCredentials

# Authenticate and create the PyDrive client.
# This only needs to be done once in a notebook.
auth.authenticate_user()
gauth = GoogleAuth()
gauth.credentials = GoogleCredentials.get_application_default()
drive = GoogleDrive(gauth)

def authenticate():
    if gauth.credentials is None:
        gauth.credentials = GoogleCredentials.get_application_default()
    elif gauth.access_token_expired:
        # Refresh them if expired
        gauth.Refresh()
    else:
        # Initialize the saved creds
        gauth.Authorize()

    drive = GoogleDrive(gauth)


def upload_best():
    authenticate()
    if os.path.isfile('./checkpoints/checkpoint_best.pt'):
        folder_id = '1tjvgBWhm9Quzy1yfnX0AOJZz55N5YZGj' # NPMT folder ID
        f = drive.CreateFile({"parents": [{"kind": "drive#fileLink", "id": folder_id}], 'title': 'checkpoint_best.pt', 'id': '1MGtk-Kiy-vGyE5aUjgt4RZoP0zObod80'})
        f.SetContentFile('./checkpoints/checkpoint_best.pt')
        f.Upload()
        print('Uploaded best checkpoint with ID {}'.format(f.get('id')))

def upload_last():
    authenticate()
    if os.path.isfile('./checkpoints/checkpoint_last.pt'):
        folder_id = '1tjvgBWhm9Quzy1yfnX0AOJZz55N5YZGj' # NPMT folder ID
        f = drive.CreateFile({"parents": [{"kind": "drive#fileLink", "id": folder_id}], 'title': 'checkpoint_last.pt', 'id': '1VqDatgr2uFRsy8Ei92OjQqFMup1sCLz0'})
        f.SetContentFile('./checkpoints/checkpoint_last.pt')
        f.Upload()
        print('Uploaded last checkpoint with ID {}'.format(f.get('id')))