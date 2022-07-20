import dlib
import cv2 as cv
import numpy as np
from PIL import Image
from PIL import ImageEnhance, ImageTk
import math
import tkinter as tk
import tkinter.filedialog
import tkinter.messagebox
import windnd
import image_to_numpy


def resize(w, h, w_box, h_box, pil_image):
    f1 = w_box / w
    f2 = h_box / h
    factor = min([f1, f2])
    width = int(w * factor)
    height = int(h * factor)
    return pil_image.resize((width, height), Image.ANTIALIAS)


def cv2pil(pic):
    # opencv对象转换为pil对象
    return Image.fromarray(cv.cvtColor(pic, cv.COLOR_RGB2BGR))


def pil2cv(pic):
    # pil对象转换为opencv对象
    return cv.cvtColor(np.asarray(pic), cv.COLOR_RGB2BGR)


class DataCenter:
    def __init__(self):
        self.img = None
        self.template = None
        self.result = None
        self.gray_face = None
        self.flag = 1
        self.lb_face = tk.Label()
        self.lb_bottom_layer = tk.Label()

    def dragged_files(self, files):
        filename = '\n'.join((item.decode('gbk') for item in files))  # 拖入图片
        if filename != '':
            try:
                image = Image.fromarray(image_to_numpy.load_image_file(filename))  # 防止导入图片时方向转向(图片转了90度变横了)
            except:
                tkinter.messagebox.showerror('错误', '您选的可能不是图片')
            self.img = pil2cv(image)
            w_box = 300
            h_box = 300
            w, h = image.size
            resize_photo = resize(w, h, w_box, h_box, image)
            photo = ImageTk.PhotoImage(resize_photo)
            lb_pic.config(image=photo)
            lb_pic.image = photo

    def pic_select(self):
        filename = tkinter.filedialog.askopenfilename()  # 同上
        if filename != '':
            try:
                image = Image.fromarray(image_to_numpy.load_image_file(filename))  # 同上
            except:
                tkinter.messagebox.showerror('错误', '您选的可能不是图片')
            self.img = pil2cv(image)
            w_box = 300
            h_box = 300
            w, h = image.size
            resize_photo = resize(w, h, w_box, h_box, image)
            photo = ImageTk.PhotoImage(resize_photo)
            lb_pic.config(image=photo)
            lb_pic.image = photo

    def template_select1(self):
        image = Image.open('attachFiles/pandaHead.png')
        self.template = image
        resize_photo = image.resize((300, 300))
        photo = ImageTk.PhotoImage(resize_photo)
        lb_template.config(image=photo)
        lb_template.image = photo

    def template_select2(self):
        image = Image.open('attachFiles/rabbit.jpg')
        self.template = image.resize((640, 640))
        resize_photo = image.resize((300, 300))
        photo = ImageTk.PhotoImage(resize_photo)
        lb_template.config(image=photo)
        lb_template.image = photo

    def template_select3(self):
        image = Image.open('attachFiles/mushroom.jpg')
        self.template = image.resize((640, 640))
        resize_photo = image.resize((300, 300))
        photo = ImageTk.PhotoImage(resize_photo)
        lb_template.config(image=photo)
        lb_template.image = photo

    def template_select4(self):
        filename = tkinter.filedialog.askopenfilename()
        if filename != '':
            image = Image.open(filename)
            w_box = 300
            h_box = 300
            w, h = image.size
            self.template = resize(w, h, 640, 640, image)
            resize_photo = resize(w, h, w_box, h_box, image)
            photo = ImageTk.PhotoImage(resize_photo)
            lb_template.config(image=photo)
            lb_template.image = photo

    def make_gray_face(self):
        gray = cv.cvtColor(self.img, cv.COLOR_BGR2GRAY)
        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor('attachFiles/shape_predictor_68_face_landmarks.dat')
        res = detector(gray, 1)

        shape = predictor(self.img, res[0])
        face = shape.parts()

        points = []
        point1 = [int((face[0].x + face[17].x) / 2), int((face[0].y + face[17].y) / 2)]
        point2 = [int((face[1].x + face[36].x) / 2), int((face[1].y + face[36].y) / 2)]
        point3 = [int((face[5].x + face[48].x) / 2), int((face[5].y + face[48].y) / 2)]
        point4 = [int((face[6].x + face[59].x) / 2), int((face[6].y + face[59].y) / 2)]
        point5 = [int((face[7].x + face[58].x) / 2), int((face[7].y + face[58].y) / 2)]
        point6 = [int((face[8].x + face[57].x) / 2), int((face[8].y + face[57].y) / 2)]
        point7 = [int((face[9].x + face[56].x) / 2), int((face[9].y + face[56].y) / 2)]
        point8 = [int((face[10].x + face[55].x) / 2), int((face[10].y + face[55].y) / 2)]
        point9 = [int((face[11].x + face[54].x) / 2), int((face[11].y + face[54].y) / 2)]
        point10 = [int((face[15].x + face[45].x) / 2), int((face[15].y + face[45].y) / 2)]
        point11 = [int((face[16].x + face[26].x) / 2), int((face[16].y + face[26].y) / 2)]
        for pos in face[17:27]:
            points.append([pos.x, pos.y])
        points.append(point11)
        points.append(point10)
        points.append(point9)
        points.append(point8)
        points.append(point7)
        points.append(point6)
        points.append(point5)
        points.append(point4)
        points.append(point3)
        points.append(point2)
        points.append(point1)

        # 脸部提取
        face_pos = np.array(points, np.int32)
        mask = np.zeros(self.img.shape, np.uint8)
        mask = cv.polylines(mask, [face_pos], True, (255, 255, 255))
        mask = cv.fillPoly(mask, [face_pos], (255, 255, 255))
        mask = cv.bitwise_and(mask, self.img)

        # 脸部旋转
        def cal_ang(point_1, point_2, point_3):
            # 三点计算夹角，返回角度对应各点的角度
            a = math.sqrt((point_2[0] - point_3[0]) * (point_2[0] - point_3[0]) + (point_2[1] - point_3[1]) * (
                    point_2[1] - point_3[1]))
            b = math.sqrt((point_1[0] - point_3[0]) * (point_1[0] - point_3[0]) + (point_1[1] - point_3[1]) * (
                    point_1[1] - point_3[1]))
            c = math.sqrt((point_1[0] - point_2[0]) * (point_1[0] - point_2[0]) + (point_1[1] - point_2[1]) * (
                    point_1[1] - point_2[1]))
            A = math.degrees(math.acos((a * a - b * b - c * c) / (-2 * b * c)))
            B = math.degrees(math.acos((b * b - a * a - c * c) / (-2 * a * c)))
            C = math.degrees(math.acos((c * c - a * a - b * b) / (-2 * a * b)))
            return math.ceil(A), math.ceil(B), math.ceil(C)

        mask = cv2pil(mask)
        angle = cal_ang((face[30].x, face[30].y), (face[30].x + 10, face[30].y), (face[27].x, face[27].y))
        ang = (90 + angle[0]) if face[30].y < face[27].y else (90 - angle[0])
        mask = mask.rotate(ang, expand=True)

        # 图像转为png格式
        mask = pil2cv(mask)
        gray = cv.cvtColor(mask, cv.COLOR_BGR2GRAY)
        binary = cv.threshold(gray, 1, 255, cv.THRESH_BINARY)[1]
        contours = cv.findContours(binary, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)[0]
        max_con = sorted(contours, key=cv.contourArea, reverse=True)[0]
        x, y, w, h = cv.boundingRect(max_con)
        face = mask[y:(y + h), x:(x + w)]
        face = cv.cvtColor(face, cv.COLOR_BGR2GRAY)
        face = cv.cvtColor(face, cv.COLOR_GRAY2BGR)

        face = cv2pil(face)
        f_gray = face.convert('L')
        face = face.convert('RGBA')
        f_data = face.getdata()
        g_data = f_gray.getdata()

        new_data = []
        for i in range(g_data.size[0] * g_data.size[1]):
            if g_data[i] == 0:
                new_data.append((f_data[i][0], f_data[i][1], f_data[i][2], 0))
            else:
                new_data.append((f_data[i][0], f_data[i][1], f_data[i][2], 255))
        face.putdata(new_data)
        return face

    def convert(self):
        if self.flag == 1:
            self.flag *= -1
            self.lb_face.config(bg='lightgray')
        else:
            self.flag *= -1
            self.lb_face.config(bg='white')

    def top_on(self):
        def image_position_x(vx):
            if var2.get() == 'A':
                var2.set('B')
                mode_select()
            self.lb_face.place(x=vx, y=vy.get(), anchor='center')

        def image_position_y(vy):
            if var2.get() == 'A':
                var2.set('B')
                mode_select()
            self.lb_face.place(x=vx.get(), y=vy, anchor='center')

        def image_size(vSize):
            if var2.get() == 'A':
                var2.set('B')
                mode_select()
            w, h = self.gray_face.size
            gray_face_resize = resize(w, h, int(vSize), int(vSize), self.gray_face)
            contrast = ImageEnhance.Contrast(gray_face_resize)
            gray_face_resize = contrast.enhance(vc.get())
            gray_face_resize = gray_face_resize.rotate(-vr.get(), expand=True)
            gray_face_resize_tk = ImageTk.PhotoImage(gray_face_resize)
            self.lb_face.config(image=gray_face_resize_tk)
            self.lb_face.image = gray_face_resize_tk

        def image_rotation(vr):
            if var2.get() == 'A':
                var2.set('B')
                mode_select()
            w, h = self.gray_face.size
            gray_face_resize = resize(w, h, vSize.get(), vSize.get(), self.gray_face)
            contrast = ImageEnhance.Contrast(gray_face_resize)
            gray_face_resize = contrast.enhance(vc.get())
            rot = gray_face_resize.rotate(-int(vr), expand=True)
            rot_tk = ImageTk.PhotoImage(rot)
            self.lb_face.config(image=rot_tk)
            self.lb_face.image = rot_tk

        def image_contrast(vc):
            if var2.get() == 'A':
                var2.set('B')
                mode_select()
            w, h = self.gray_face.size
            temp = resize(w, h, vSize.get(), vSize.get(), self.gray_face)
            temp = temp.rotate(-vr.get(), expand=True)
            contrast = ImageEnhance.Contrast(temp)
            temp = contrast.enhance(float(vc))
            temp_tk = ImageTk.PhotoImage(temp)
            self.lb_face.config(image=temp_tk)
            self.lb_face.image = temp_tk

        def para_set(vx_value, vy_value, vSize_value, vr_value, vc_value):
            # 参数设置
            vx.set(vx_value)
            s_x.set(vx_value)
            image_position_x(vx_value)
            vy.set(vy_value)
            s_y.set(vy_value)
            image_position_y(vy_value)
            vSize.set(vSize_value)
            s_size.set(vSize_value)
            image_size(vSize_value)
            vr.set(vr_value)
            s_rotation.set(vr_value)
            image_rotation(vr_value)
            vc.set(vc_value)
            s_contrast.set(vc_value)
            image_contrast(vc_value)

        def result_make_auto():
            # 自动模式制作结果
            contrast = ImageEnhance.Contrast(self.gray_face)
            face = contrast.enhance(3.6)
            # 模板需要准备的参数
            if var.get() == 'A':
                e_x, e_y, e_w, e_h, e_angle = (190, 160, 250, 250, 0)
            elif var.get() == 'B':
                e_x, e_y, e_w, e_h, e_angle = (200, 205, 240, 240, 0)
            elif var.get() == 'C':
                e_x, e_y, e_w, e_h, e_angle = (170, 200, 230, 230, 0)
            else:
                e_x, e_y, e_w, e_h, e_angle = (190, 160, 250, 250, 0)
            face = face.rotate(e_angle, expand=True)
            fw, fh = face.size
            face = face.resize((int(fw / (fh / e_h)), e_h), Image.ANTIALIAS)
            fw, fh = face.size
            face = face.crop((int((fw - e_w) / 2), 0, int((fw - e_w) / 2) + e_w, fh))
            self.result = self.template.copy()
            self.result.paste(face, (e_x, e_y, e_x + e_w, e_y + e_h), mask=face.split()[-1])
            self.result = self.result.resize((300, 300))

        def result_make_hand():
            # 手动模式制作结果
            w, h = self.gray_face.size
            gray_face_resize = resize(w, h, vSize.get(), vSize.get(), self.gray_face)
            contrast = ImageEnhance.Contrast(gray_face_resize)
            gray_face_resize = contrast.enhance(vc.get())
            gray_face_resize = gray_face_resize.rotate(-vr.get(), expand=True)
            w, h = gray_face_resize.size
            self.result = self.template.copy()
            self.result = resize(self.result.size[0], self.result.size[1], 300, 300, self.result)
            self.result.paste(gray_face_resize, (int(vx.get() - w / 2) - 35, int(vy.get() - h / 2) - 35,
                                                 int(vx.get() - w / 2) - 35 + w, int(vy.get() - h / 2) - 35 + h),
                              mask=gray_face_resize.split()[-1])

        def mode_select():
            if var2.get() == 'A':
                result_make_auto()
                result_tk = ImageTk.PhotoImage(self.result)
                self.lb_face.place_forget()
                self.lb_bottom_layer.config(image=result_tk)
                self.lb_bottom_layer.image = result_tk
            else:
                template_tk = ImageTk.PhotoImage(resize(self.template.size[0], self.template.size[1], 300, 300, self.template))
                self.lb_bottom_layer.config(image=template_tk)
                self.lb_bottom_layer.image = template_tk
                para_set(vx.get(), vy.get(), vSize.get(), vr.get(), vc.get())

        def preview():
            second = tk.Toplevel()
            second.title('预览')
            if var2.get() == 'B':
                result_make_hand()
            result_tk = ImageTk.PhotoImage(self.result)
            lb_result_show = tk.Label(second)
            lb_result_show.config(image=result_tk)
            lb_result_show.image = result_tk
            lb_result_show.pack()

        def pic_save():
            if var2.get() == 'B':
                result_make_hand()
            save_path = tk.filedialog.asksaveasfilename(title=u'保存文件', filetypes=[("PNG", ".png")])
            self.result.save('{}.png'.format(save_path))
            tk.messagebox.showinfo('成功！', '保存成功！')

        top = tk.Toplevel()
        top.geometry('870x530')
        top.title('操作面板')
        top.resizable(False, False)
        c_top_bg = tk.Canvas(top, bg='white', height=618, width=1000)
        c_top_bg.create_rectangle(30, 30, 340, 340)
        c_top_bg.pack()
        try:
            self.gray_face = data.make_gray_face()
        except:
            tkinter.messagebox.showerror('错误', '出错啦，可能原因：\n1、未选择图片或图片无脸\n2、未选择模板')

        template_tk = ImageTk.PhotoImage(resize(self.template.size[0], self.template.size[1], 300, 300, self.template))
        self.lb_bottom_layer = tk.Label(top, image=template_tk)
        self.lb_bottom_layer.place(x=185, y=185, anchor='center')

        gray_face_tk = ImageTk.PhotoImage(self.gray_face)
        self.lb_face = tk.Label(top, image=gray_face_tk, bg='white')
        self.lb_face.place(x=200, y=200, anchor='center')

        var2 = tk.StringVar()
        var2.set('A')
        r_auto = tk.Radiobutton(top, text='自动', font='宋体', variable=var2, value='A', command=mode_select)
        r_auto.place(x=155, y=370, anchor='e')
        r_manual = tk.Radiobutton(top, text='手动', font='宋体', variable=var2, value='B', command=mode_select)
        r_manual.place(x=185, y=370, anchor='w')

        vx = tk.IntVar()
        vy = tk.IntVar()
        vSize = tk.IntVar()
        vr = tk.IntVar()
        vc = tk.DoubleVar()

        s_x = tk.Scale(top, from_=0, to=300, orient=tk.HORIZONTAL, length=350, showvalue=True, variable=vx,
                       tickinterval=50, resolution=1, command=image_position_x)
        s_x.set(180)
        s_x.place(x=450, y=20)
        s_y = tk.Scale(top, from_=0, to=300, orient=tk.HORIZONTAL, length=350, showvalue=True, variable=vy,
                       tickinterval=50, resolution=1, command=image_position_y)
        s_y.set(160)
        s_y.place(x=450, y=120)
        s_size = tk.Scale(top, from_=0, to=300, orient=tk.HORIZONTAL, length=350, showvalue=True, variable=vSize,
                          tickinterval=50, resolution=1, command=image_size)
        s_size.set(max(self.gray_face.size))
        s_size.place(x=450, y=220)
        s_rotation = tk.Scale(top, from_=-180, to=180, orient=tk.HORIZONTAL, length=350, showvalue=True, variable=vr,
                              tickinterval=45, resolution=1, command=image_rotation)
        s_rotation.set(0)
        s_rotation.place(x=450, y=320)
        s_contrast = tk.Scale(top, from_=0, to=20, orient=tk.HORIZONTAL, length=350, showvalue=True, variable=vc,
                              tickinterval=4, resolution=0.01, command=image_contrast)
        s_contrast.set(3.60)
        s_contrast.place(x=450, y=420)

        lb_x = tk.Label(top, text='左右:', font=20, bg='white')
        lb_x.place(x=360, y=40)
        lb_y = tk.Label(top, text='上下:', font=20, bg='white')
        lb_y.place(x=360, y=140)
        lb_s = tk.Label(top, text='大小:', font=20, bg='white')
        lb_s.place(x=360, y=240)
        lb_r = tk.Label(top, text='旋转:', font=20, bg='white')
        lb_r.place(x=360, y=340)
        lb_c = tk.Label(top, text='对比度:', font=20, bg='white')
        lb_c.place(x=360, y=440)

        # 参数初始化
        if var.get() == 'A':
            para_set(184, 166, 122, 0, 4.10)
        elif var.get() == 'B':
            para_set(186, 188, 119, 0, 4.55)
        elif var.get() == 'C':
            para_set(169, 182, 114, 0, 4.55)
        else:
            para_set(184, 166, 122, 0, 4.10)

        b_save = tk.Button(top, text='保存图片', font=20, command=pic_save)
        b_save.place(x=185, y=470, anchor='w')
        b_return = tk.Button(top, text='返回', font=20, command=top.destroy)
        b_return.place(x=85, y=470, anchor='w')
        b_convert = tk.Button(top, text='查看透明区域', font=20, command=data.convert)
        b_convert.place(x=185, y=420, anchor='w')
        b_preview = tk.Button(top, text='预览', font=20, command=preview)
        b_preview.place(x=85, y=420, anchor='w')

        top.mainloop()


if __name__ == "__main__":
    window = tk.Tk()
    window.title('Panda Emoji Maker')
    window.geometry('800x494')
    window.resizable(False, False)
    c_bg = tk.Canvas(window, bg='white', height=494, width=800)  # 白色背景
    c_bg.create_rectangle(50, 50, 350, 350)  # 绘制矩形(x1,y1,x2,y2)，选照片的边框
    c_bg.create_rectangle(450, 50, 750, 350)  # 绘制矩形(x1,y1,x2,y2)，选模板的边框
    c_bg.pack()
    c_bg.create_rectangle(360, 190, 440, 210, fill='black')  # 绘制加号
    c_bg.create_rectangle(390, 160, 410, 240, fill='black')  # 绘制加号

    data = DataCenter()

    b_pic_select = tk.Button(window, text='选择照片', font=('宋体', 15), command=data.pic_select)
    b_pic_select.place(x=200, y=390, anchor="center")

    lb_pic2 = tk.Label(window, text='请选择您的绝世容颜', font=30, bg='white')
    lb_pic2.place(x=200, y=200, anchor="center")
    lb_pic = tk.Label(window, text='请选择您的绝世容颜', font=30, bg='white')
    lb_pic.place(x=200, y=200, anchor="center")
    windnd.hook_dropfiles(lb_pic, func=data.dragged_files)  # 拖入图片
    lb_template = tk.Label(window, text='请选择您的熊猫头模板', font=30, bg='white')
    lb_template.place(x=600, y=200, anchor='center')

    var = tk.StringVar()
    r1_template_select = tk.Radiobutton(window, text='熊猫头', font='宋体', variable=var, value='A', command=data.template_select1)
    r1_template_select.place(x=560, y=370, anchor='center')
    r2_template_select = tk.Radiobutton(window, text='小白兔', font='宋体', variable=var, value='B', command=data.template_select2)
    r2_template_select.place(x=650, y=370, anchor='center')
    r3_template_select = tk.Radiobutton(window, text='蘑菇头', font='宋体', variable=var, value='C', command=data.template_select3)
    r3_template_select.place(x=560, y=400, anchor='center')
    r4_template_select = tk.Radiobutton(window, text='自己选', font='宋体', variable=var, value='D', command=data.template_select4)
    r4_template_select.place(x=650, y=400, anchor='center')

    b_make = tk.Button(window, text='制作表情', font=('黑体', 20), command=data.top_on)  # 打开操作面板
    b_make.place(x=400, y=450, anchor='center')

    window.mainloop()
