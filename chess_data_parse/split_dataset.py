import json

import torch


def split_dataset(player_path):
    games = torch.load(f"./dataset_2/{player_path}", weights_only=True)

    white_game_list = games["white"]
    black_game_list = games["black"]

    # First, filter out games with fewer than 30 moves.
    min_moves = 30
    valid_white_games = [g for g in white_game_list if g['state'].size(0) >= min_moves]
    valid_black_games = [g for g in black_game_list if g['state'].size(0) >= min_moves]

    # If there are more than 32 games, select the longest 30 games.
    max_allowed = 32
    select_count = 30
    if len(valid_white_games) > max_allowed:
        valid_white_games = sorted(valid_white_games, key=lambda g: g['state'].size(0), reverse=True)[:max_allowed]
    if len(valid_black_games) > max_allowed:
        valid_black_games = sorted(valid_black_games, key=lambda g: g['state'].size(0), reverse=True)[:max_allowed]

    print(len(valid_white_games), len(valid_black_games))

    new_player_files = {
        "white": valid_white_games,
        "black": valid_black_games
    }

    torch.save(new_player_files, f"./dataset/{player_path.split('/')[-1].split('.')[0]}.pt")
    print(f"Saved {player_path.split('.')[0]}.pt")


if __name__ == '__main__':
    with open("processed_players.json", "r", encoding="utf-8") as f:
        player_files = json.load(f)

    for file in player_files:
        path = player_files[file]
        split_dataset(path.split("/")[-1])
