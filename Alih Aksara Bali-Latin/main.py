from enum import auto
from tkinter import *
import tkinter.font
from tkinter import filedialog
from function import identification
from PIL import ImageTk, Image

# Set Window
root = Tk()
root.geometry("750x500")
root.title("Sistem Alih Aksara Bali")
h1 = tkinter.font.Font(size=20)
Label(root, text="SISTEM ALIH AKSARA BALI", font=h1).pack()

# Heading Template

# Open File Style
filename = ''
def chooseFile ():
  global filename, canvas
  filename = filedialog.askopenfilename(filetypes = (("Text files", "*.png"), ("all files", "*.*")))
  img = Image.open(filename)
  img = img.resize((300,150), Image.ANTIALIAS)
  img = ImageTk.PhotoImage(img)
  lbl.configure(image = img, height=200)
  lbl.image = img

  print('t' + filename)
  Button(root, text = "Alih Aksara", command=lambda: identification(result, pathImage=filename), font="Normal 15").pack()
  
  result = Label(root, text="", wraplength=300)
  result.pack()

# DISPLAY
chooseFile = Button(root, text='Open', command=chooseFile)
chooseFile.pack()
print('t' + filename)
lbl = Label(root)
lbl.pack()

root.mainloop()