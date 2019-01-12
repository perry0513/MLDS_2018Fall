def add_arguments(parser):
    '''
    Add your arguments here if needed. The TAs will run test.py to load
    your default arguments.

    For example:
        parser.add_argument('--batch_size', type=int, default=32, help='batch size for training')
        parser.add_argument('--learning_rate', type=float, default=0.01, help='learning rate for training')
    '''
    parser.add_argument('--batch_size', type=int, default=1, help='batch size for training')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='learning rate for training')
    parser.add_argument('--episodes', type=int, default=5000, help='episodes for training')
    parser.add_argument('--save_history_period', type=int, default=10, help='save current reward in training per period')
    parser.add_argument('--load_checkpoint', action='store_true')
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--discount_factor', type=float, default=0.99)
    return parser
