import sys
import random

from PyQt5.QtWidgets import QApplication
from PyQt5.QtWidgets import QWidget
from PyQt5.QtWidgets import QPushButton
from PyQt5.QtWidgets import QLabel
from PyQt5.QtWidgets import QVBoxLayout
from PyQt5.QtWidgets import QGridLayout
from PyQt5.QtWidgets import QHBoxLayout
from PyQt5.QtWidgets import QFrame
from PyQt5.QtWidgets import QInputDialog
from PyQt5.QtCore import QPropertyAnimation
from PyQt5.QtCore import Qt
from PyQt5.QtCore import QPointF
from PyQt5.QtCore import QSequentialAnimationGroup

class Window(QWidget):
    """
    UI属性如下：
        self.order追踪块的序列
        self.labels追踪每一个块（Label控件）
        self.solutions是移动块的序列，例如[1, 2, 3]就是把块1,2,3串行地移动到空白块
        self.positions记录坐标，方便移动，它的值是不变的
        self.group是为了实现串行动画
    只需要用setSolution设置solitions属性，然后点击start就可以看到效果，注意这里不会
    检查移动的正确性，所以需要提供解的一方提供正确的解
    点击reset可以打乱所有块，且保证能够还原到目标状态，这里目标状态设定为[0-8]，可以考虑
    以后进行扩展，即达到想要的目标状态
    """
    def __init__(self):
        super().__init__()
        # 记录所有模块
        self.labels = []
        self.order = [x for x in range(9)]
        self.group = QSequentialAnimationGroup()
        self.solutions = []
        self.positions = []
        self.initUI()
    

    def initUI(self):
        self.setGeometry(400, 200, 650, 700)
        self.setWindowTitle("八数码问题")
        self.createLayout()

        self.show()
        # 追踪每个初始位置，用以坐标变换
        for each in self.labels:
            self.positions.append((each.geometry().x(), each.geometry().y()))


    def createLayout(self):
        """创建布局"""
        # 主布局
        main_layout = QVBoxLayout()
        
        # 局部布局
        h_layout = QHBoxLayout()
        grid_layout = QGridLayout()

        bt1 = QPushButton("Start")
        bt2 = QPushButton("Reset")
        bt3 = QPushButton("Random")
        bt4 = QPushButton("Test")
        bt1.setMaximumSize(100, 50)
        bt2.setMaximumSize(100, 50)
        bt3.setMaximumSize(100, 50)
        bt4.setMaximumSize(100, 50)
        h_layout.addWidget(bt1)
        h_layout.addWidget(bt2)
        h_layout.addWidget(bt3)
        h_layout.addWidget(bt4)

        bt1.clicked.connect(self.startCallback)
        bt2.clicked.connect(self.resetCallback)
        bt3.clicked.connect(self.randomButtonCallback)
        bt4.clicked.connect(self.testButtonCallback)
        
        i = 0
        for m in range(3):
            for n in range(3):
                tmp = QLabel()
                if i != 0:
                    tmp.setStyleSheet("border-image:url(images/%d.png)" % self.order[i])
                tmp.setFixedSize(200, 200)
                self.labels.append(tmp)
                grid_layout.addWidget(tmp, m, n)
                i += 1

        hwg = QWidget()
        gwg = QFrame()
        hwg.setMaximumHeight(150)
        gwg.setFrameShape(2)
        gwg.setFrameShadow(0x030)

        hwg.setLayout(h_layout)
        gwg.setLayout(grid_layout)

        main_layout.addWidget(hwg)
        main_layout.addWidget(gwg)
        self.setLayout(main_layout)


    def startCallback(self):
        self.group.clear()
        for each in self.solutions:
            self.swap(each)
        self.group.start()
    
    def resetCallback(self):
        text, ok = QInputDialog.getText(self, "Input Dialog", "请输入序列（空白块用0表示）：")
        if ok:
            self.initialQuestion(list(map(int, text)))

    def randomButtonCallback(self):
        self.group.clear()
        self.randomOrder()
        self.reload()

    def testButtonCallback(self):
        text, ok = QInputDialog.getText(self, "Input Dialog", "请输入解序列：")
        if ok:
            self.solute(list(map(int, text)))


    def swap(self, aim):
        """交换空白块和相邻块"""
        b_x, b_y = self.positions[self.order[0]]
        a_x, a_y = self.positions[self.order[aim]]
        self.order[0], self.order[aim] = self.order[aim], self.order[0]
        self.moveAnimation(self.labels[aim], (a_x, a_y), (b_x, b_y))
        self.labels[0].move(a_x, a_y)

    def moveAnimation(self, obj, start, end):
        """移动obj从坐标start到坐标end"""
        anim = QPropertyAnimation(obj, b'pos')
        anim.setDuration(1000)
        anim.setStartValue(QPointF(start[0], start[1]))
        anim.setEndValue(QPointF(end[0], end[1]))
        self.group.addAnimation(anim)

    def randomOrder(self):
        """打乱顺序，要确保能够还原"""
        random.shuffle(self.order)
        # 检查是否能拼回，即order的逆序数加上空白块的横纵坐标
        cnt = 0
        length = len(self.order)
        for i in range(1, length):
            for j in range(i):
                if self.order[j] > self.order[i]:
                    cnt += 1
        cnt += self.order[0] // 4 + self.order[0] % 4

        # 对不可通关情况处理，即调换非空白块
        if cnt % 2:
            self.order[1], self.order[2] = self.order[2], self.order[1]

    def reload(self):
        """重新加载内容"""
        length = len(self.labels)
        for i in range(length):
            self.labels[i].move(self.positions[self.order[i]][0], self.positions[self.order[i]][1])
        

    def initialQuestion(self, order):
        """初始化问题，order为输入的初始序列"""
        cnt = 0
        for each in order:
            self.order[each] = cnt
            cnt += 1
        self.reload()

    def solute(self, sol):
        """求解问题，sol为求解序列"""
        self.solutions = sol
        self.startCallback()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    wnd = Window()
    sys.exit(app.exec_())
