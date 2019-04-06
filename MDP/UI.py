from tkinter import *

def getColor(value):
    if abs(value) > 1:
        value = int(value)
    clr = hex(int(abs(200*value)))[2:]
    if len(clr) == 1:
        clr = '0' + clr
    clr = ('#'+clr+'0000') if value < 0 else ('#00'+clr+'00')
    return clr

def drawGrid(canvas, canvas_width, canvas_height, gridSize, margin, reward, maze):
    m, n = -1, -1
    for x in range(margin, canvas_width - margin, gridSize):
        m += 1
        n = -1
        for y in range(margin, canvas_height - margin, gridSize):
            n += 1
            if maze[n][m] != 'e':
                canvas.create_rectangle(x, y, x + gridSize, y + gridSize, 
                            fill="#888", outline="#009688", width=3)
            else:
                clr = getColor(reward[n][m])
                textPos = (x + gridSize / 2, y + gridSize / 2)
                canvas.create_rectangle(x, y, x + gridSize, y + gridSize, 
                            fill=clr, outline="#009688", width=3)
                canvas.create_rectangle(x + margin, y + margin, x + gridSize - margin, y + gridSize - margin, 
                            fill=clr, outline="#009688", width=3)
                canvas.create_text(*textPos, fill='#fff', text="%.2f" % reward[n][m], font=('Times', 20, 'bold'))

def drawTriangle(canvas, canvas_width, canvas_height, gridSize, margin, qvalues, policy, maze):
    m, n = -1, -1
    for x in range(margin, canvas_width - margin, gridSize):
        m += 1
        n = -1
        for y in range(margin, canvas_height - margin, gridSize):
            n += 1
            if maze[n][m] == 'w' or maze[n][m] == 'e':
                continue

            center = x + gridSize / 2, y + gridSize / 2
            corners = [(x, y), (x + gridSize, y), 
                    (x + gridSize, y + gridSize), (x, y + gridSize)]
            
            for i in range(4):
                clr = getColor(qvalues[(n, m)][i])
                textClr = '#fff' if i == policy[(n, m)] else '#777'
                points = [*corners[i], *corners[(i+1)%4], *center]
                canvas.create_polygon(points, outline="#009688",
                            fill=clr, width=3)
                if i % 2 == 0:
                    textPos = (corners[i][0] + corners[(i+1)%4][0]) / 2, (corners[i][1] + center[1]) / 2
                else:
                    textPos = (corners[i][0] + center[0]) / 2, (corners[i][1] + corners[(i+1)%4][1]) / 2
                canvas.create_text(*textPos, fill=textClr, text="%.2f" % qvalues[(n, m)][i], font=('Times', 16, 'bold'))

def drawDirection(canvas, canvas_width, canvas_height, gridSize, margin, values, policy, maze):
    m, n = -1, -1
    directions = (
        (
            0, - gridSize / 2 + margin, 
            margin, - gridSize / 2 + 2 * margin, 
            - margin, - gridSize / 2 + 2 * margin
        ),
        (
            gridSize / 2 - margin, 0, 
            gridSize / 2 - 2 * margin, margin, 
            gridSize / 2 - 2 * margin, - margin
        ),
        (
            0, gridSize / 2 - margin, 
            margin, gridSize / 2 - 2 * margin, 
            - margin, gridSize / 2 - 2 * margin
        ),
        (
            - gridSize / 2 + margin, 0, 
            - gridSize / 2 + 2 * margin, margin, 
            - gridSize / 2 + 2 * margin, - margin
        ),
    )
    for x in range(margin, canvas_width - margin, gridSize):
        m += 1
        n = -1
        for y in range(margin, canvas_height - margin, gridSize):
            n += 1
            if maze[n][m] == 'w' or maze[n][m] == 'e':
                continue

            clr = getColor(values[(n, m)])
            center = x + gridSize / 2, y + gridSize / 2
            points = []
            for i in range(0, 6, 2):
                points.append(center[0] + directions[policy[(n, m)]][i])
                points.append(center[1] + directions[policy[(n, m)]][i+1])
            
            canvas.create_rectangle(x, y, x + gridSize, y + gridSize, 
                            fill=clr, outline="#009688", width=3)
            canvas.create_text(*center, fill='#fff', text="%.2f" % values[(n, m)], 
                            font=('Times', 20, 'bold'))
            canvas.create_polygon(points,
                            fill='#fff', width=3)            
