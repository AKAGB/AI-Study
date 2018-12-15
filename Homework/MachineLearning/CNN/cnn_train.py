from cs231n.data_utils import get_CIFAR10_data
from cs231n.classifiers.mycnn import CNN
from cs231n.solver import Solver

dataset = get_CIFAR10_data()

train_data = {
    'X_train': dataset['X_train'],
    'y_train': dataset['y_train'],
    'X_val'  : dataset['X_val'],
    'y_val'  : dataset['y_val'],
}


model = CNN()

solver = Solver(model, train_data, 
                update_rule='adam',
                optim_config={
                    'learning_rate': 0.001,
                },
                lr_decay=0.95,
                num_epochs=50, batch_size=100,
                print_every=100)

solver.train()

solver.check_accuracy(dataset['X_test'], dataset['y_test'])