#include <iostream>
#include <limits>
#include <stdlib.h>

using namespace std;

int max_depth = 10;



enum Element {
	WHITE = -1, SPACE, BLACK
};

struct Action {
	pair<int, int> pos;
	int score = 0;					// 执行该动作的估计分数
};

struct boardEle {
	enum Element color;					// 棋子颜色
	int stable = 0;						// 记录吃掉的棋子数
};

struct State {
	boardEle board[6][6];		// 记录棋盘
	Element player;					// 下一个落子的玩家
	int black;					// 棋盘上的黑子数
	int white;					// 棋盘上的白子数

	State() {
		for (int i = 0; i < 6; i++)
			for (int j = 0; j < 6; j++)
				board[i][j].color = SPACE;
		player = BLACK;
		black = 0;
		white = 0;
	}

	void calc_stable() {
		for (int x = 0; x < 6; x++) {
			for (int y = 0; y < 6; y++) {
				if (board[x][y].color == SPACE) {
					for (int i = -1; i <= 1; i++) {
						for (int j = -1; j <= 1; j++) {
							if (i || j) {
								// 8个方向
								int num = 0;
								int i2, j2;
								for (i2 = x + i, j2 = y + j; i2 >= 0 && i2 <= 5 && j2 >= 0 && j2 <= 5; i2 += i, j2 += j) {
									// 沿着那个方向走，如果遇到己方棋子则该位置合理
									if (board[i2][j2].color == (0 - player)) {
										num++;
									}
									else if (board[i2][j2].color == SPACE) {
										num = 0;
										break;
									}
									else {
										break;
									}
								}
								// 保证在界内
								if (i2 >= 0 && i2 <= 5 && j2 >= 0 && j2 <= 5)
									board[x][y].stable += num;
							}
						}
					}
				}
			}
		}
	}

	// 返回当前状态可执行的动作，结尾处添加(0,0)表示终止
	Action *Actions() {
		return NULL;
	}

	// 返回执行了action动作后的新节点
	State *Result(Action *action) {
		return NULL;
	}

	void show() {
		cout << "\n  ";
		for (int i = 0; i<6; i++)
		{
			cout << "  " << i + 1;
		}
		cout << "\n    ─────────────────\n";
		for (int i = 0; i<6; i++)
		{
			cout << i + 1 << "--│";
			for (int j = 0; j<6; j++)
			{
				switch (board[i][j].color)
				{
				case BLACK:
					cout << "○│";
					break;
				case WHITE:
					cout << "●│";
					break;
				case SPACE:
					if (board[i][j].stable)
					{
						cout << " " << board[i][j].stable <<"│";
					}
					else
					{
						cout << "  │";
					}
					break;
				default:    /* 棋子颜色错误 */
					cout << "* │";
				}
			}
			cout << "\n    ─────────────────\n";
		}

		cout << ">>>白棋(●)个数为:" << white << "         ";
		cout << ">>>黑棋(○)个数为:" << black << endl << endl << endl;
	}
};

/**
  * state: 搜索的初始状态
  * return: 返回执行的动作
  */
//Action *Alpha_Beta_Search(State *state);


/**
  * state: 输入的状态
  * alpha: 评估值下界
  * beta: 评估值上界
  * depth: 当前深度
  * return: 返回当前节点的代价(MAX节点)
  */
//Action *max_value(State *state, int alpha, int beta, int depth);

/**
  * state: 输入的状态
  * alpha: 评估值下界
  * beta: 评估值上界
  * depth: 当前深度
  * return: 返回当前节点的代价(MIN节点)
  */
//Action *min_value(State *state, int alpha, int beta, int depth);

/**
  * state: 当前状态
  * depth: 当前节点的深度
  * return: 如果depth大于最大搜索深度或state为叶子节点则返回true，反之返回false
  */
// bool cutoff_test(State *state, int depth);

/**
  * state: 当前状态
  * return: 评估值
  */
// int eval(State *state);

int main()
{
	State *root = new State();
	root->board[2][2].color = BLACK;
	root->board[3][3].color = BLACK;
	root->board[2][3].color = WHITE;
	root->board[3][2].color = WHITE;
	root->calc_stable();
	root->show();

	system("pause");
	return 0;
}
/*
Action *Alpha_Beta_Search(State *state) {
	return max_value(state, INT_MIN, INT_MAX, 0);
}

Action *max_value(State *state, int alpha, int beta, int depth) {
	if (cutoff_test(state, depth)) {
		int tmp = eval(state);
		Action *result = new Action();
		result->pos.first = result->pos.second = 0;
		result->score = tmp;
		return result;
	}
		
	int v = INT_MIN;
	Action *actions = state->Actions();
	int cnt = 0;
	while (actions[cnt].pos.first != 0) {
		State *next_state = state->Result(&actions[cnt]);
		Action *tmp = min_value(next_state, alpha, beta, depth+1);
		actions[cnt].score = tmp->score;
		if (tmp->score > v) v = tmp->score;
		if (v >= beta) {
			Action *result = new Action();
			result->pos = actions[cnt].pos;
			result->score = v;
			delete[] actions;
			return result;
		}
		if (v > alpha) alpha = v;
		cnt++;
		delete next_state;
	}
	for (int i = 0; i < cnt; i++) {
		if (v == actions[i].score) {
			Action *result = new Action();
			result->pos = actions[i].pos;
			result->score = v;
			delete[] actions;
			return result;
		}
	}
	
}

Action *min_value(State *state, int alpha, int beta, int depth) {
	if (cutoff_test(state, depth)) {
		int tmp = eval(state);
		Action *result = new Action();
		result->pos.first = result->pos.second = 0;
		result->score = tmp;
		return result;
	}

	int v = INT_MAX;
	Action *actions = state->Actions();
	int cnt = 0;
	while (actions[cnt].pos.first != 0) {
		State *next_state = state->Result(&actions[cnt]);
		Action *tmp = max_value(next_state, alpha, beta, depth + 1);
		actions[cnt].score = tmp->score;
		if (tmp->score < v) v = tmp->score;
		if (v <= alpha) {
			Action *result = new Action();
			result->pos = actions[cnt].pos;
			result->score = v;
			delete[] actions;
			return result;
		}
		if (beta > v) beta = v;
		cnt++;
		delete next_state;
	}
	for (int i = 0; i < cnt; i++) {
		if (v == actions[i].score) {
			Action *result = new Action();
			result->pos = actions[i].pos;
			result->score = v;
			delete[] actions;
			return result;
		}
	}
}
*/