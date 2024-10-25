from tkinter import *
from tkinter import ttk
import math
import subprocess
import numpy

pixels = {}

def motion_detected(event):
    event_x = event.x
    event_y = event.y

    lenght = 40
    x = math.floor(event_x / 20)
    y = math.floor(event_y / 20)
    pixels[(y * 28) + x] = {255}
    pixels[(y * 28) + x + 1] = {255}
    pixels[((y + 1) * 28) + x] = {255}
    pixels[((y + 1) * 28) + x + 1] = {255}
    x0 = x * 20 
    y0 = y * 20
    x1 = x * 20 + lenght
    y1 = y * 20 + lenght
    
    canvas.create_rectangle(x0, y0, x1, y1, fill="white", outline="white")

def clear(event):
    canvas.delete("all")
    return

def elaborate(event):
    if(event.char == chr(13)):
        f = open("C:\\Users\\Gabriele\\Desktop\\VS Work Dir\\Progetti C++\\Primi Test\\MLP\\digit.csv", "w")
        n_image = blurring()
        i = 0
        while(i < 784):
            f.write(str(n_image.get(i, 0)).replace("{", "").replace("}", ""))
            if(i != 783):
                f.write(",")
            i += 1
        f.close()
        clear(event)
        pixels.clear()
        subprocess.call(["C:\\Users\\Gabriele\\Desktop\\VS Work Dir\\Progetti C++\\Primi Test\\MLP\\Sources\\execute.bat"])
        r = open("C:\\Users\\Gabriele\\Desktop\\VS Work Dir\\Progetti C++\\Primi Test\\MLP\\result.txt", "r")
        canvas.create_text(30, 30, text="Cifra: " + str(r.read()), fill="white")

def blurring():
    m = [0.075, 0.075, 0.075, 0.075, 0.4, 0.075, 0.075, 0.075, 0.075]
    n_image = {}
    i = 0
    while(i < 784):
        pi = [int(str(pixels.get(i - 29, 0)).replace("{", "").replace("}", "")), int(str(pixels.get(i - 28 , 0)).replace("{", "").replace("}", "")), int(str(pixels.get(i - 27, 0)).replace("{", "").replace("}", "")), int(str(pixels.get(i - 1, 0)).replace("{", "").replace("}", "")), int(str(pixels.get(i, 0)).replace("{", "").replace("}", "")), int(str(pixels.get(i + 1, 0)).replace("{", "").replace("}", "")), int(str(pixels.get(i + 27, 0)).replace("{", "").replace("}", "")), int(str(pixels.get(i + 28, 0)).replace("{", "").replace("}", "")), int(str(pixels.get(i + 29, 0)).replace("{", "").replace("}", ""))]
        c = numpy.convolve(pi, m, "same")
        j = 0
        n_image[i] = round(c[4])
        i += 1
    return n_image


root = Tk()
#frm = ttk.Frame(root, padding=10)
#frm.grid()
#ttk.Button(frm, text="Quit", command=root.destroy).grid(column=1, row=0)
canvas = Canvas (root, width=559, height=559, bg="black")
canvas.pack()
canvas.bind("<B1-Motion>", motion_detected)
root.bind("<Button-3>", clear)
root.bind("<Key>", elaborate)
root.mainloop()
