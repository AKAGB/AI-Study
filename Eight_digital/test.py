def calc_evaluation(order):
    """启发式函数为曼哈顿距离"""
    aim = [ (0, 0), (0, 1), (0, 2), 
            (0, 3), (1, 0), (1, 1), (1, 2), 
            (1, 3), (2, 0), (2, 1), (2, 2), 
            (2, 3), (3, 0), (3, 1), (3, 2), (3, 3)]
    result = 0
    for i in range(16):
        if order[i]:
            result += abs(aim[order[i]-1][0]-aim[i][0]) + abs(aim[order[i]-1][1]-aim[i][1])
    return result
    

a = [9, 6, 3, 4, 15, 1, 7, 8, 14, 0, 11, 10, 2, 5, 13, 12]
print(calc_evaluation(a))