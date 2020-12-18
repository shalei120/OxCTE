import os
import tkinter as tk
import tkinter.filedialog as fd
from tkinter import Label
import copy
class AnnoUI:
    def __init__(self):
        self.rootdir = './artifacts/'

    def loadData(self):
        self.origin_data = []
        with open(self.rootdir + 'AnnoTestData.txt', 'r') as wh:
            for line in wh.readlines():
                title, raw_content, change_content, raw_context = line.strip().split('\t')
                self.origin_data.append([title, raw_content, change_content, raw_context])
            wh.close()

        self.Complete_data = []
        index = -1
        if os.path.exists(self.rootdir + 'LabeledTestData.txt'):
            with open(self.rootdir + 'LabeledTestData.txt', 'r') as wh:
                index = wh.readline()
                index = int(index.strip())
                for ind , line in enumerate(wh.readlines()):
                    title, raw_content, change_content, raw_context, Dprime = line.strip().split('\t')
                    self.Complete_data.append([title, raw_content, change_content, raw_context, Dprime])
                wh.close()
        else:
            index = 0
            self.Complete_data = copy.copy(self.origin_data)
            for ind in range(len(self.Complete_data)):
                self.Complete_data[ind].append(self.Complete_data[ind][3])
        # nextdata = len(self.Complete_data)
        self.badnum = len([1 for title, raw_content, change_content, raw_context, Dprime in self.Complete_data if change_content == 'bad'])
        return index

    def showcase(self, index):
        self.cur_index = index
        if index < 0 or index >= len(self.origin_data):
            return
        title, raw_content, change_content, raw_context, Dp = self.Complete_data[index]
        self.D_str.set(raw_context)
        self.Q_str.set(title)
        self.A_str.set(raw_content)
        # self.Ap_str.set(change_content)
        self.Ap_str = change_content
        self.Ap_label.delete(1.0,tk.END)
        self.Ap_label.insert(1.0, self.Ap_str)
        self.txt_edit.delete(1.0, tk.END)
        self.txt_edit.insert(1.0, Dp)
        self.cur_index_strvar.set(str(self.cur_index))
        self.bad_str.set('Bad: ' + str(self.badnum)+ '\nTotal:' + str(len(self.Complete_data)))
        # canvas

    def saveData(self):
        with open(self.rootdir + 'LabeledTestData.txt', 'w') as wh:
            wh.write(str(self.cur_index))
            wh.write('\n')
            for title, raw_content, change_content, raw_context, Dprime in self.Complete_data:
                wh.write('\t'.join([title, raw_content, change_content, raw_context, Dprime]))
                wh.write('\n')
            wh.close()

    def saveCurindex(self):
        new_ap = self.Ap_label.get(1.0, tk.END).strip()
        new_Dp = self.txt_edit.get(1.0, tk.END).strip()
        # assert self.cur_index == len(self.Complete_data)
        if new_ap != self.Ap_str:
            self.Complete_data[self.cur_index][2] = new_ap
        if new_Dp != self.D_str.get().strip():
            self.Complete_data[self.cur_index][4] = new_Dp
        return new_ap

    def lastcase(self):
        new_ap = self.saveCurindex()
        if new_ap == 'bad':
            self.badnum += 1
        self.showcase(self.cur_index - 1)

    def nextcase(self):
        new_ap = self.saveCurindex()
        if new_ap == 'bad':
            self.badnum += 1
        self.showcase(self.cur_index + 1)

    def main(self):
        """主函数：设置窗口部件，指定按钮点击事件处理函数
        """
        self.cur_index = self.loadData()
        window = tk.Tk()
        window.geometry("1000x800")
        window.title("Data Grader")
        self.D_str = tk.StringVar()
        self.D_str.set("")
        self.Q_str = tk.StringVar()
        self.Q_str.set("")
        self.A_str = tk.StringVar()
        self.A_str.set("")
        # self.Ap_str = tk.StringVar()
        # self.Ap_str.set("")
        self.Ap_str = ''

        self.D_label =Label(window, wraplength=600, textvariable= self.D_str)
        self.D_label.grid(column=0, row=0)
        self.Q_label =Label(window, textvariable= self.Q_str)
        self.Q_label.grid(column=0, row=1)
        self.A_label =Label(window, textvariable= self.A_str)
        self.A_label.grid(column=0, row=2)
        self.Ap_label =tk.Text(window, height = 5, width = 50)#, textvariable= self.Ap_str)
        self.Ap_label.grid(column=0, row=3)
        self.bad_str = tk.StringVar()
        self.bad_str.set('Bad: ' + str(self.badnum) + '\nTotal:' + str(len(self.Complete_data)))
        self.bad_label = Label(window, textvariable= self.bad_str)#, textvariable= self.Ap_str)
        self.bad_label.grid(column=1, row=0)
        self.txt_edit = tk.Text(window,height = 10, width = 50)
        self.cur_index_strvar = tk.StringVar()
        self.cur_index_strvar.set(str(self.cur_index))
        self.curindexLabel = Label(window, textvariable= self.cur_index_strvar)

        self.D_label.config(font=("Calibri", 22))
        self.Q_label.config(font=("Calibri", 22))
        self.A_label.config(font=("Calibri", 22))
        self.Ap_label.config(font=("Courier", 22))
        self.txt_edit.config(font=("Courier", 22))
        self.curindexLabel.config(font=("Calibri", 22))
        # canvas = tk.Canvas(window, width=1000, height=700)
        # canvas.pack(side="top")
        button = tk.Button(window, text="Save",
                           command=lambda: self.saveData(),height = 5, width = 20)
        left = tk.Button(window, text="<=",
                           command=lambda: self.lastcase(),height = 5, width = 20)
        right = tk.Button(window, text="=>",
                           command=lambda: self.nextcase(),height = 5, width = 20)
        self.showcase(self.cur_index)
        self.curindexLabel.grid(column=1, row=1)
        self.txt_edit.grid(column=0, row=4)
        button.grid(column=1, row=4)
        left.grid(column=1, row=2)
        right.grid(column=1, row=3)
        # self.D_label.pack()
        # self.Q_label.pack()
        # self.A_label.pack()
        # self.Ap_label.pack()
        # button.pack()
        # left.pack()
        # right.pack()
        tk.mainloop()


if __name__ == "__main__":
    a = AnnoUI()
    a.main()
