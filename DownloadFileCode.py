import requests
import os
#from tkinter import filedialog
#from tkinter import *
url = 'http://google.com/favicon.ico'
r = requests.get(url, allow_redirects=True)

#root = Tk()
#root.filename = filedialog.asksaveasfilename(initialdir="/", title="Select file",
#                                             filetypes=(("jpeg files", "*.jpg"), ("png files", "*.png"),  ("all files", "*.*")))
#print(root.filename)
open('output_img.png', 'wb').write(r.content)
print('File saved in ' + os.getcwd( ))

