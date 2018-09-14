import math
from enum import Enum

class Result(Enum):
    Found = 1
    Not_Found = 2

class State:
    """
    状态类：记录每一个结点的状态
    """
    aim = [(x, y) for x in range(4) for y in range(4)]

    def __init__(self, depth=0, order=(), num=0):
        self.order = order[:]           # 记录块顺序，下标是块位置，元素是块类型，注意和UI中的order相反
        self.depth = depth              # 当前深度，即走到当前状态所需要的代价
        self.hn = 0                     # 到目标状态的最小估计值
        self.fn = 0                     # 从起始状态到目标状态的最小估计值
        self.num = num                  # 产生该动作移动的块
        self.calc_evaluation()

    def __lt__(self, other):
        """重载小于号，方便堆的操作"""
        return self.fn < other.fn

    def __eq__(self, other):
        return self.order == other.order

    def calc_evaluation(self):
        """启发式函数为曼哈顿距离"""
        order = self.order[:]
        result = 0
        for i in range(16):
            if order[i]:
                cur = (order[i] - 1) // 4, (order[i] - 1) % 4
                result += abs(cur[0]-State.aim[i][0]) + abs(cur[1]-State.aim[i][1])
        self.hn = result
        self.fn = self.depth + self.hn


def successors(node):
    result = []
    actions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    idx = node.order.index(0)
    for action in actions:
        x, y = idx // 4, idx % 4
        if (0 <= x + action[0] <= 3) and (0 <= y + action[1] <= 3):
            order = list(node.order)
            aim = order[idx + action[0]*4+action[1]]
            order[idx], order[idx + action[0]*4+action[1]] = order[idx + action[0]*4+action[1]], order[idx]
            tmp = State(node.depth+1, tuple(order), aim)
            result.append(tmp)
    return sorted(result)

def is_goal(node):
    # goal = list(range(1, 16)) + [0]
    return node.order == (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 0)

def cost(node, succ):
    return 1

def ida_star(root):
    bound = root.hn
    path = [root]
    hash_table = set()
    hash_table.add(root.order)
    while True:
        print(bound)
        t = search(path, hash_table, 0, bound)
        if t == Result.Found:
            return path, bound
        if t == math.inf:
            return Result.Not_Found
        bound = t

def search(path, hash_table, g, bound):
    node = path[-1]
    if node.fn > bound:
        return node.fn
    if is_goal(node):
        return Result.Found
    min_cost = math.inf
    succs = successors(node)
    for succ in succs:
        if succ.order not in hash_table:
            path.append(succ)
            hash_table.add(succ.order)
            t = search(path, hash_table, g + 1, bound)
            if t == Result.Found:
                return Result.Found
            if t < min_cost:
                min_cost = t
            tmp = path.pop()
            hash_table.remove(tmp.order)
    return min_cost


if __name__ == '__main__':
    # 56:
    # problem = [11, 3, 1, 7, 4, 6, 8, 2, 15, 9, 10, 13, 14, 12, 5, 0]
    # problem = [1, 3, 14, 4, 7, 11, 0, 6, 5, 9, 8, 12, 13, 10, 15, 2]
    # problem = [1, 2, 3, 4, 5, 7, 0, 8, 9, 6, 10, 12, 13, 14, 11, 15]
    problem = (9, 6, 3, 4, 15, 1, 7, 8, 14, 0, 11, 10, 2, 5, 13, 12)
    # 48:problem = [6, 10, 3, 15, 14, 8, 7, 11, 5, 1, 0, 2, 13, 12, 9, 4]
    # 49:problem = (14, 10, 6, 0, 4, 9, 1, 8, 2, 3, 5, 11, 12, 13, 7, 15)
    # 62:problem = [0, 5, 15, 14, 7, 9, 6, 13, 1, 2, 12, 10, 8, 11, 4, 3]
    root = State(0, problem, 0)
    solution = ida_star(root)
    if solution != Result.Not_Found:
        print('Total:', solution[1])
        for each in solution[0][1:]:
            print(each.num, end=' ')
    else:
        print('No solution')
