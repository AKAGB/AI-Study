import heapq

class Solution:
    """利用A*算法求解八数码问题的解决类"""
    def __init__(self):
        self.frointer = []              # 记录叶子节点
        self.explored = []              # 记录已经添加过的状态，实现图搜索

    def A_star_search(self, input_str):
        """A*搜索"""
        self.frointer = []
        self.explored = []
        actions = ['w', 's', 'a', 'd']
        order = list(map(int, input_str))
        root = State(order[:])
        heapq.heappush(self.frointer, root)
        while True:
            if len(self.frointer) == 0:
                return False
            node = heapq.heappop(self.frointer)
            if self.goalTest(node.order):
                return self.getSolution(node)
            self.explored.append(node.order)
            for each in actions:
                child = self.childNode(node, each)
                if child:
                    if (child.order not in self.explored) and (child not in self.frointer):
                        heapq.heappush(self.frointer, child)
                    elif child in self.frointer:
                        idx = self.frointer.index(child)
                        if child < self.frointer[idx]:
                            del self.frointer[idx]
                            heapq.heappush(self.frointer, child)

    def childNode(self, parent, action):
        new_order = self.result(parent.order[:], action)
        if new_order:
            return State(new_order, action, parent)
        return None

    def result(self, order, action):
        """order根据action产生新的order，若出现出界则返回None"""
        idx = order.index(0)
        if action == 'w' and idx - 3 >= 0:
            order[idx-3], order[idx] = order[idx], order[idx-3]
        elif action == 's' and idx + 3 <= 8:
            order[idx+3], order[idx] = order[idx], order[idx+3]
        elif action == 'a' and idx % 3 - 1 >= 0:
            order[idx-1], order[idx] = order[idx], order[idx-1]
        elif action == 'd' and idx % 3 + 1 <= 2:
            order[idx+1], order[idx] = order[idx], order[idx+1]
        else:
            return None
        return order

    def goalTest(self, order):
        goal_test = [0, 1, 2, 3, 4, 5, 6, 7, 8]
        return goal_test == order

    def getSolution(self, node):
        result = ''
        while node:
            result = node.action + result
            node = node.parent
        return result


class State:
    """
    状态类：记录每一个结点的状态
    """
    def __init__(self, order=[], action='', parent=None):
        self.order = order[:]           # 记录块顺序，下标是块位置，元素是块类型，注意和UI中的order相反
        self.action = action            # 生成该状态的行为
        self.depth = 0                  # 当前深度，即走到当前状态所需要的代价
        self.gn = 0                     # 到目标状态的最小估计值
        self.hn = 0                     # 从起始状态到目标状态的最小估计值
        self.parent = parent            # 记录父节点
        self.calc_evaluation()

    def __lt__(self, other):
        """重载小于号，方便堆的操作"""
        return self.hn < other.hn

    def __eq__(self, other):
        return self.order == other.order

    def calc_evaluation(self):
        """启发式函数为曼哈顿距离"""
        result = 0
        length = len(self.order)
        for i in range(length):
            if self.order[i] != 0:
                cur = self.order[i] // 3, self.order[i] % 3
                aim = i // 3, i % 3
                result += abs(cur[0]-aim[0]) + abs(cur[1]-aim[1])
        self.gn = result
        if self.parent:
            self.depth = self.parent.depth + 1
        self.hn = self.depth + self.gn

if __name__ == '__main__':
    sol = Solution()
    result = sol.A_star_search('724506831')
    print(result)
