# import random
# from collections import deque
# from mcts import MCTS
# 
# def generate_self_play(game_cls, net, mcts_simulations):
#     memory = []
#     game = game_cls()
#     mcts = MCTS(game_cls(), net, mcts_simulations)
# 
#     while not game.is_terminal():
#         visits = mcts.run(game)
#         actions, counts = zip(*visits.items())
#         probs = [c / sum(counts) for c in counts]
#         action = random.choices(actions, probs)[0]
# 
#         memory.append((game.encode_tensor(), probs, None))  # reward未知
#         game = game.play(action)
# 
#     result = game.reward()
#     memory = [(s, p, result) for (s, p, _) in memory]
#     return memory