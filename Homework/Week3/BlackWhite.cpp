#include <iostream>
#include <limits>
#include <stdlib.h>

using namespace std;

int max_depth = 10;

int V[6][6] = {
	{ 90, -60, 10, 10, -60, 90 },
	{ -60, -80, 5, 5, -80, -60 },
	{ 10, 5, 1, 1, 5, 10 },
	{ 10, 5, 1, 1, 5, 10 },
	{ -60, -80, 5, 5, -80, -60 },
	{ 90, -60, 10, 10, -60, 90 }
};


enum Element {
	WHITE = -1, SPACE, BLACK
};

struct Action {
	pair<int, int> pos;
	int score = 0;					// Record the evaluation value of action
};

struct boardEle {
	enum Element color;					// color of chessman
	int stable = 0;						// stable value
};

struct State {
	boardEle board[6][6];			// Record board
	Element player;					// The next player
	int black;						// Number of black
	int white;						// Number of white

	State() {
		for (int i = 0; i < 6; i++)
			for (int j = 0; j < 6; j++)
				board[i][j].color = SPACE;
		player = BLACK;
		black = 0;
		white = 0;
	}

	State(State &other) {
		for (int i = 0; i < 6; i++)
			for (int j = 0; j < 6; j++) {
				board[i][j].color = other.board[i][j].color;
				board[i][j].stable = 0;
			}

		player = other.player;
		black = other.black;
		white = other.white;
	}

	// Calculate the stable number of SPACE
	void calc_stable() {
		for (int x = 0; x < 6; x++) {
			for (int y = 0; y < 6; y++) {
				board[x][y].stable = 0;
				if (board[x][y].color == SPACE) {
					for (int i = -1; i <= 1; i++) {
						for (int j = -1; j <= 1; j++) {
							if (i || j) {
								// 8 directions
								int num = 0;
								int i2, j2;
								for (i2 = x + i, j2 = y + j; i2 >= 0 && i2 <= 5 && j2 >= 0 && j2 <= 5; i2 += i, j2 += j) {
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
								if (i2 >= 0 && i2 <= 5 && j2 >= 0 && j2 <= 5)
									board[x][y].stable += num;
							}
						}
					}
				}
			}
		}
	}

	// Calculate number of white and black
	void calc_num() {
		white = black = 0;
		for (int i = 0; i < 6; i++) {
			for (int j = 0; j < 6; j++) {
				if (board[i][j].color == BLACK)
					black++;
				else if (board[i][j].color == WHITE)
					white++;
			}
		}
	}

	// return all actions after current state (end with <0, 0>)
	Action *Actions() {
		calc_stable();
		int cnt = 0;
		for (int i = 0; i < 6; i++)
			for (int j = 0; j < 6; j++)
				if (board[i][j].stable) cnt++;

		Action *result = new Action[cnt + 1];
		int k = 0;

		for (int i = 0; i < 6; i++) {
			for (int j = 0; j < 6; j++) {
				if (board[i][j].stable) {
					result[k++].pos = pair<int, int>(i + 1, j + 1);
				}
			}
		}
		result[cnt].pos = pair<int, int>(0, 0);
		return result;
	}

	// return new state with action
	State *Result(Action *action) {
		State *result = new State(*this);
		int x = action->pos.first - 1, y = action->pos.second - 1;
		if (board[x][y].stable) {
			result->board[x][y].color = result->player;
			result->board[x][y].stable = 0;
			for (int i = -1; i <= 1; i++) {
				for (int j = -1; j <= 1; j++) {
					if (i || j) {
						int num = 0;
						int i2, j2;
						for (i2 = x + i, j2 = y + j; i2 >= 0 && i2 <= 5 && j2 >= 0 && j2 <= 5; i2 += i, j2 += j) {
							if (board[i2][j2].color == (0 - result->player)) {
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
						if (i2 >= 0 && i2 <= 5 && j2 >= 0 && j2 <= 5 && num != 0) {
							i2 -= i;
							j2 -= j;
							while (i2 != x || j2 != y) {
								result->board[i2][j2].color = (Element)(0 - result->board[i2][j2].color);
								i2 -= i;
								j2 -= j;
							}
						}
					}
				}
			}
			result->player = (Element)(0 - player);
			result->calc_stable();
			result->calc_num();
			return result;
		}
		else {
			return NULL;
		}

	}

	// Calculate the number of stable point
	void Stable() {
		for (int i = 0; i < 6; i++) {
			for (int j = 0; j < 6; j++) {
				if (board[i][j].color != SPACE) {
					board[i][j].stable = 1;
					for (int x = -1; x <= 1; x++) {
						for (int y = -1; y <= 1; y++) {
							if (x == 0 && y == 0) {
								x = y = 2;
							}
							else {
								int flag = 2;
								for (int i2 = i + x, j2 = j + y; i2 >= 0 && i2 <= 5 && j2 >= 0 && j2 <= 5; i2 += x, j2 += y)
								{
									if (board[i2][j2].color != board[i][j].color)
									{
										flag--;
										break;
									}
								}

								for (int i2 = i - x, j2 = j - y; i2 >= 0 && i2 <= 5 && j2 >= 0 && j2 <= 5; i2 -= x, j2 -= y)
								{
									if (board[i2][j2].color != board[i][j].color)
									{
										flag--;
										break;
									}
								}

								if (flag)    
								{
									board[i][j].stable++;
								}
							}
						}
					}
				}
			}
		}
	}

	// show layout
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
						cout << " +│";
					}
					else
					{
						cout << "  │";
					}
					break;
				default:   
					cout << "* │";
				}
			}
			cout << "\n    ─────────────────\n";
		}

		cout << ">>>Number of WHITE(●):" << white << "         ";
		cout << ">>>Number of BLACK(○):" << black << endl << endl << endl;
	}
};

/**
* state: Current state
* return: Action with Evaluation value
*/
Action *Alpha_Beta_Search(State *state);


/**
* state: Current state
* alpha: Lower bound of evaluation value
* beta: Upper bound of evaluation value
* depth: Current depth
* return: Action with Evaluation value
*/
Action *max_value(State *state, int alpha, int beta, int depth);

/**
* state: Current state
* alpha: Lower bound of evaluation value
* beta: Upper bound of evaluation value
* depth: Current depth
* return: Action with Evaluation value
*/
Action *min_value(State *state, int alpha, int beta, int depth);

/**
* state: Current state
* return: Evaluation value
*/
int eval(State *state);

int main()
{
	State *root = new State();
	root->board[2][2].color = WHITE;
	root->board[3][3].color = WHITE;
	root->board[2][3].color = BLACK;
	root->board[3][2].color = BLACK;
	root->calc_stable();
	root->calc_num();
	root->show();
	system("pause");

	while (1) {
		int x, y;
		if (root->player == BLACK) {
            // Human player
			cout << "Black Player：" << endl;
			Action *ac = root->Actions();
			if (ac->pos.first == 0) {
				cout << "Passing" << endl;
				root->player = WHITE;
				continue;
			}
			cout << "Input new pos: ";
			cin >> x >> y;
			Action action;
			action.pos = pair<int, int>(x, y);
			State *next_state = root->Result(&action);
			if (next_state) {
				next_state->show();
				delete root;
				root = next_state;
			}
			else
				cout << "Invalid Position, Please input again: " << endl;
			system("pause");
		}
		else {
            // AI Player
			cout << "White Player：" << endl;
			Action *action = Alpha_Beta_Search(root);
			if (action->pos.first != 0) {
				cout << "Position: <" << action->pos.first << ", " << action->pos.second << ">" << endl;
				cout << "score: " << action->score << endl;
				State *next_state = root->Result(action);
				if (next_state) {
					next_state->show();
					delete root;
					root = next_state;
				}
				else
					cout << "Invalid Position, Please input again: " << endl;
			}
			else {
				cout << "Game Over" << endl;
				if (root->black > root->white)
					cout << "Black Player Win!" << endl;
				else if (root->black < root->white)
					cout << "White Player Win!" << endl;
				else
					cout << "Draw" << endl;
				break;
			}
			delete action;
			//system("pause");
		}

	}

	return 0;
}

Action *Alpha_Beta_Search(State *state) {
	// White player min the value, black to max
	if (state->player == WHITE)
		return min_value(state, INT_MIN, INT_MAX, 0);
	else
		return max_value(state, INT_MIN, INT_MAX, 0);
}

Action *max_value(State *state, int alpha, int beta, int depth) {
	if (depth > max_depth) {
		// limit depth
		int tmp = eval(state);
		Action *result = new Action();
		result->pos.first = result->pos.second = 0;
		result->score = tmp;
		return result;
	}

	// Alpha Pruning
	int v = INT_MIN;
	Action *actions = state->Actions();
	int cnt = 0;
	while (actions[cnt].pos.first != 0) {
		State *next_state = state->Result(&actions[cnt]);
		Action *tmp = min_value(next_state, alpha, beta, depth + 1);
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
		delete tmp;
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
	// This player can not perform, check opponent
	if (cnt == 0) {
		state->player = (Element)(0 - state->player);
		Action *actions = state->Actions();
		if (actions[0].pos.first != 0) {
			// opponent can perform
			Action *tmp = min_value(state, alpha, beta, depth + 1);
			tmp->pos = pair<int, int>(0, 0);
			return tmp;
		}
		else {
			Action *result = new Action();
			result->pos = pair<int, int>(0, 0);
			// Both player can not perform
			if (state->black > state->white)
				result->score = INT_MAX - 1;
			else if (state->black < state->white)
				result->score = INT_MIN + 1;
			else
				result->score = 0;
			return result;
		}
	}
}

Action *min_value(State *state, int alpha, int beta, int depth) {

	if (depth > max_depth) {
		// limit depth
		int tmp = eval(state);
		Action *result = new Action();
		result->pos = pair<int, int>(0, 0);
		result->score = tmp;
		return result;
	}

	// Beta Pruning
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
		delete tmp;
		delete next_state;
	}
	// return action with min score v
	for (int i = 0; i < cnt; i++) {
		if (v == actions[i].score) {
			Action *result = new Action();
			result->pos = actions[i].pos;
			result->score = v;
			delete[] actions;
			return result;
		}
	}
	// This player can not perform, check opponent
	if (cnt == 0) {
		state->player = (Element)(0 - state->player);
		Action *actions = state->Actions();
		if (actions[0].pos.first != 0) {
			// opponent can perform
			Action *tmp = max_value(state, alpha, beta, depth + 1);
			tmp->pos = pair<int, int>(0, 0);
			return tmp;
		}
		else {
			// Both player can not perform
			Action *result = new Action();
			result->pos = pair<int, int>(0, 0);
			
			if (state->black < state->white)
				result->score = INT_MIN + 1;
			else if (state->black > state->white)
				result->score = INT_MAX - 1;
			else
				result->score = 0;
			return result;
		}
	}
}

int eval(State *state) {
	state->Stable();
	int value = 0;
	// Calculate the stable value
	for (int i = 0; i<6; i++)
	{
		for (int j = 0; j<6; j++)
		{
			value += (state->board[i][j].color)*(state->board[i][j].stable);
		}
	}

	// stable value should be more important
	value *= 2;

	// Calculate the position value
	for (int i = 0; i<6; i++)
	{
		for (int j = 0; j<6; j++)
		{
			value += V[i][j] * state->board[i][j].color;
		}
	}

	return value;
}