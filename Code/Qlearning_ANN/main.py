from IPython.display import clear_output
import Code.Qlearning_ANN.Qlearning as Qlearning

Qlearning = Qlearning.Qlearning()

while(1):

    print('Please select what you want to do:\n')
    print('1- User controlled snake\n')
    print('2- Qlearning training\n')
    print('3- Qlearning game visualization\n')
    print('4- ANN training\n')
    print('5- ANN game visualization\n')

    action = input('Please select what you want to do:')

    clear_output(wait=True)

    if action == '1':
        pass
    elif action == '2':
        Qlearning.QlearningTraining()
    elif action == '3':
        Qlearning.startGameQlearning()
    elif action == '4':
        pass
    elif action == '5':
        pass
    else:
        print('Invalid action')
        continue