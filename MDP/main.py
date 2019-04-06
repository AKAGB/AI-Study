from agent import *
from UI import *

def shift(event, data):
    data['ValueDisplay'] = not data['ValueDisplay']
    if data['ValueDisplay']:
        drawDirection(data['w'], data['canvas_width'], data['canvas_height'], 
                    data['gridSize'], data['margin'], data['agent'].values, 
                    data['agent'].getPolicy(), data['maze'])
    else:
        drawTriangle(data['w'], data['canvas_width'], data['canvas_height'], 
                    data['gridSize'], data['margin'], data['agent'].qvalues, 
                    data['agent'].getPolicy(), data['maze'])

maze = [
    '000e',
    '0w0e',
    '0000'
]
reward = [
    [-0.03, -0.03, -0.03, 1],
    [-0.03, -0.03, -0.03, -1],
    [-0.03, -0.03, -0.03, -0.03]
]

mdp = MDP(maze, reward)

# Get Input
opt = input('Please select agent(1 -- Value Iteration Agent | 2 -- Policy Iteration Agent): ')
iteration = int(input('Please enter the iterations: '))
if opt == '1':
    agent = ValueIterationAgent(mdp, 0.9, iteration)
else:
    agent = PolicyIterationAgent(mdp, 0.9, iteration)

data = {
    'ValueDisplay': True,
    'canvas_width': 620,
    'canvas_height': 470,
    'gridSize': 150,
    'margin': 10,
    'reward': reward,
    'maze': maze,
    'agent': agent
}
# Show Result
master = Tk()

master.title('After %d Iterations' % iteration)
w = Canvas(master,
        width=data['canvas_width'],
        height=data['canvas_height'])
w.pack()
data['w'] = w
drawGrid(w, data['canvas_width'], data['canvas_height'], 150, 10, reward, maze)

master.bind('<Return>', lambda x : shift(x, data))

drawDirection(data['w'], data['canvas_width'], data['canvas_height'], 
                data['gridSize'], data['margin'], data['agent'].values, 
                data['agent'].getPolicy(), data['maze'])


mainloop()