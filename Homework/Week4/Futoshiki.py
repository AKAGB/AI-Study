import sys

class Problem:
    def __init__(self, puzzle, cons, limit):
        self.puzzle = puzzle
        self.size = len(puzzle)
        self.cons = cons
        self.dom = []
        self.max_depth = limit
        
        
    def curDom(self, x):
        """Return Curdom of x"""
        if self.puzzle[x[0]][x[1]]:
            return [self.puzzle[x[0]][x[1]]]
        result = list(range(1, self.size+1))
        
        for each in self.cons:
            if x == each[0] and self.puzzle[each[1][0]][each[1][1]]:
                result = list(filter(lambda n:n<self.puzzle[each[1][0]][each[1][1]], result))
            elif x == each[1] and self.puzzle[each[0][0]][each[0][1]]:
                result = list(filter(lambda n:n>self.puzzle[each[0][0]][each[0][1]], result))

        for i in range(self.size):
            # row where x in
            if self.puzzle[x[0]][i]:
                result = list(filter(lambda n:n!=self.puzzle[x[0]][i], result))
            # colomn where x in
            if self.puzzle[i][x[1]]:
                result = list(filter(lambda n:n!=self.puzzle[i][x[1]], result))
        return result

    def calc_dom(self):
        for i in range(self.size):
            self.dom.append([])
            for j in range(self.size):
                self.dom[i].append(self.curDom((i, j)))

    def FCCheck(self, x):
        # According "<" or ">" to limit the curdom of X
        order_dom = list(range(1, self.size+1))
        for each in self.cons:
            if x == each[0] and self.puzzle[each[1][0]][each[1][1]]:
                order_dom = list(filter(lambda n:n<self.puzzle[each[1][0]][each[1][1]], order_dom))
            elif x == each[1] and self.puzzle[each[0][0]][each[0][1]]:
                order_dom = list(filter(lambda n:n>self.puzzle[each[0][0]][each[0][1]], order_dom))

        for d in self.dom[x[0]][x[1]][:]:
            if d not in order_dom:
                self.dom[x[0]][x[1]].remove(d)
                continue
            # check row and line
            flag = False
            for i in range(self.size):
                if self.puzzle[x[0]][i] == d or self.puzzle[i][x[1]] == d:
                    flag = True
                    break
            if flag:
                self.dom[x[0]][x[1]].remove(d)
        if self.dom[x[0]][x[1]]:
            return True
        return False

    def FC(self, level):
        # print(level)
        if level >= self.max_depth:
            # Successful
            # print(level)
            print_puzzle(self.puzzle)
            sys.exit()

        V = self.pickVar()
        for d in self.dom[V[0]][V[1]]:
            # Remmember dom
            dom = []
            for i in range(self.size):
                dom.append([])
                for j in range(self.size):
                    dom[i].append(self.dom[i][j][:])

            self.puzzle[V[0]][V[1]] = d
            DWOoccured = False
            # Check all variable in the same row and column.
            for i in range(self.size):
                if self.puzzle[V[0]][i] == 0 and self.FCCheck((V[0], i)) == False:
                    DWOoccured = True
                    break
                if self.puzzle[i][V[1]] == 0 and self.FCCheck((i, V[1])) == False:
                    DWOoccured = True
                    break
            if not DWOoccured:
                self.FC(level + 1)
            # Restore all values pruned by FCCheck
            self.dom = dom
            self.puzzle[V[0]][V[1]] = 0
            
        # Assigned[V] = False
        

    def pickVar(self):
        result = (-1, -1)
        for i in range(self.size):
            for j in range(self.size):
                if self.puzzle[i][j] == 0:
                    if result[0] == -1:
                        result = (i, j)
                    elif len(self.dom[result[0]][result[1]]) > len(self.dom[i][j]):
                        result = (i, j)
        return result

    
def print_puzzle(puzzle):
    for each in puzzle:
        print(each)


if __name__ == '__main__':
    puzzle = []
    for each in range(9):
        puzzle.append([0]*9)
    # puzzle[2][1] = 4
    # C = [   ((0, 2), (0, 3)), ((1, 1), (0, 1)), ((2, 1), (2, 0)),
    #         ((2, 3), (2, 2)), ((2, 1), (3, 1)), ((3, 2), (2, 2)),
    #         ((3, 3), (2, 3)), ((4, 0), (3, 0)), ((4, 2), (3, 2)),
    #         ((4, 1), (4, 0))]
    puzzle[0][3] = 7
    puzzle[0][4] = 3
    puzzle[0][5] = 8
    puzzle[0][7] = 5
    puzzle[1][2] = 7
    puzzle[1][5] = 2
    puzzle[2][5] = 9
    puzzle[3][3] = 4
    puzzle[4][2] = 1
    puzzle[4][6] = 6
    puzzle[4][7] = 4
    puzzle[5][6] = 2
    puzzle[8][8] = 6
    # Constraints, each (x, y) means puzzle(x) < puzzle(y)
    C = [   ((0, 0), (0, 1)), ((0, 3), (0, 2)), ((1, 3), (1, 4)),
            ((1, 6), (1, 7)), ((2, 6), (1, 6)), ((2, 1), (2, 0)),
            ((2, 2), (2, 3)), ((2, 3), (3, 3)), ((3, 3), (3, 2)),
            ((3, 5), (3, 4)), ((3, 5), (3, 6)), ((3, 8), (3, 7)),
            ((4, 1), (3, 1)), ((4, 5), (3, 5)), ((4, 0), (4, 1)),
            ((5, 4), (4, 4)), ((5, 8), (4, 8)), ((5, 1), (5, 2)),
            ((5, 4), (5, 5)), ((5, 7), (5, 6)), ((5, 1), (6, 1)),
            ((6, 6), (5, 6)), ((6, 8), (5, 8)), ((6, 3), (6, 4)),
            ((7, 7), (6, 7)), ((7, 1), (8, 1)), ((8, 2), (7, 2)),
            ((7, 5), (8, 5)), ((8, 8), (7, 8)), ((8, 5), (8, 6))
        ]
    pb = Problem(puzzle, C, 81-13)
    pb.calc_dom()
    pb.FC(0)
    # print(pb.FCCheck((0, 2)))
    # print(pb.dom[0][2])
