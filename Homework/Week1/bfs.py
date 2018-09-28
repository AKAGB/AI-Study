
class Node:
    def __init__(self, state, action, cost, parent=None):
        self.state = state
        self.action = action
        self.cost = cost
        self.parent = parent

class Maze:
    def __init__(self, maze):
        self.maze = maze
        line = 0
        for each in maze:
            start = each.find('S')
            end = each.find('E')
            if start != -1:
                init_state = [line, start]
            if end != -1:
                goal_state = [line, end]
            line += 1
        self.init_state = init_state
        self.goal_state = goal_state

    def Goal_test(self, cur):
        return cur == self.goal_state

    def Actions(self, node):
        """Optional actions"""
        result = []
        if node.state[0] + 1 <= len(self.maze) and \
            self.maze[node.state[0]+1][node.state[1]] != '1':
            result.append('s')
        if node.state[0] - 1 >= 0 and \
            self.maze[node.state[0]-1][node.state[1]] != '1':
            result.append('w')
        if node.state[1] + 1 <= len(self.maze[0]) and \
            self.maze[node.state[0]][node.state[1]+1] != '1':
            result.append('d')
        if node.state[1] - 1 >= 0 and \
            self.maze[node.state[0]][node.state[1]-1] != '1':
            result.append('a')
        return result

    def Result(self, node, action):
        """Return new state"""
        if action == 's':
            return [node.state[0]+1, node.state[1]]
        if action == 'w':
            return [node.state[0]-1, node.state[1]]
        if action == 'd':
            return [node.state[0], node.state[1]+1]
        if action == 'a':
            return [node.state[0], node.state[1]-1]

    def StepCost(self, parent, action):
        return 1

def Solution(node):
    """Return solution"""
    sol = []
    while node:
        sol.insert(0, node.state)
        node = node.parent
    return sol

def Child_node(problem, parent, action):
    """Return new node"""
    state = problem.Result(parent, action)
    cost = parent.cost + problem.StepCost(parent, action)
    return Node(state, action, cost, parent)

def BreadthFirstSearch(problem):
    """BFS"""
    node = Node(problem.init_state, None, 0)
    frontier = [node]
    explored = []
    if problem.Goal_test(node.state):
        return Solution(node)
    while True:
        if len(frontier) == 0:
            # No solution
            return None
        node = frontier.pop(0)
        explored.append(node.state)
        actions = problem.Actions(node)
        for action in actions:
            child = Child_node(problem, node, action)
            if (child.state not in explored) and (child not in frontier):
                if problem.Goal_test(child.state):
                    return Solution(child)
                frontier.append(child)

if __name__ == '__main__':
    maze = []
    with open('maze.txt', 'r') as maze_file:
        for each_line in maze_file:
            maze.append(each_line.strip())
    problem = Maze(maze)
    solution = BreadthFirstSearch(problem)
    print(solution)
