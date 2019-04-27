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


folder_id = '1GQIkxkuKoPZOgEaWI3bMs-802g8Uu9tC' # NPMT folder ID


def upload_best():
    authenticate()
    if os.path.isfile('./checkpoints/checkpoint_best.pt'):
        f = drive.CreateFile({"parents": [{"kind": "drive#fileLink", "id": folder_id}], 'title': 'checkpoint_best.pt', 'id': '1N7nvpaByCjpt5AM5I6E6bLtchRvIhVx_'})
        f.SetContentFile('./checkpoints/checkpoint_best.pt')
        f.Upload()
        print('Uploaded best checkpoint with ID {}'.format(f.get('id')))

def upload_last():
    authenticate()
    if os.path.isfile('./checkpoints/checkpoint_last.pt'):
        f = drive.CreateFile({"parents": [{"kind": "drive#fileLink", "id": folder_id}], 'title': 'checkpoint_last.pt', 'id': '1ECfL4zqBe-Mw2xJoOAu6SEe3tkXAu_t6'})
        f.SetContentFile('./checkpoints/checkpoint_last.pt')
        f.Upload()
        print('Uploaded last checkpoint with ID {}'.format(f.get('id')))
