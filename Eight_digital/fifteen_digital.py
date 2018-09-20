import math
from enum import Enum

aim = [(x, y) for x in range(4) for y in range(4)]

class Result(Enum):
    Found = 1
    Not_Found = 2

class State:
    """
    状态类：记录每一个结点的状态
    """
    
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


    def calc_evaluation(self):
        """启发式函数为曼哈顿距离"""
        order = self.order[:]
        result = 0
        for i in range(16):
            if order[i]:
                result += abs(aim[order[i]-1][0]-aim[i][0]) + abs(aim[order[i]-1][1]-aim[i][1])
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
    return node.order == (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 0)

def ida_star(root):
    bound = root.hn
    path = [root]
    hash_table = set()
    hash_table.add(root.order)
    while True:
        # print(bound)
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
    # problem = (9, 6, 3, 4, 15, 1, 7, 8, 14, 0, 11, 10, 2, 5, 13, 12)
    # problem = (6, 10, 3, 15, 14, 8, 7, 11, 5, 1, 0, 2, 13, 12, 9, 4)
    print('Please input the case: ')
    problem = []
    for i in range(4):
        data = input()
        problem.extend(list(map(int, data.split(' '))))
    root = State(0, tuple(problem), 0)
    solution = ida_star(root)
    if solution != Result.Not_Found:
        print('Lower bound:', root.hn)
        print('Total:', solution[1])
        for each in solution[0][1:]:
            print(each.num, end=' ')
    else:
        print('No solution')
