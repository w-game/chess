import os
import re
import subprocess
import chess.pgn
import concurrent.futures
from .pgn_parsering import GamesFile
import bz2
import pandas as pd

# 输入PGN文件路径和输出目录
input_pgn = "./lichess_db_standard_rated_2021-02.pgn"          # 请将此路径替换为实际的PGN文件路径
output_dir = "players_pgn_output"          # 输出目录，将存放按玩家分割的PGN及chunk数据
os.makedirs(output_dir, exist_ok=True)

def add_player(p, d):
    try:
        d[p] += 1
    except KeyError:
        d[p] = 1

def parse_elo(elo_str):
    if elo_str is None or elo_str == "?" or elo_str.strip() == "":
        return None
    match = re.match(r"(\d+)", elo_str)
    return int(match.group(1)) if match else None


def collect_players():
    games = GamesFile(input_pgn)
    counts = {}
    for i, (d, _) in enumerate(games):
        # if args.exclude_bullet and 'Bullet' in d['Event']:
        #     continue
        # else:

        white_elo = d['WhiteElo']
        black_elo = d['BlackElo']
        w_elo_val = parse_elo(white_elo)
        b_elo_val = parse_elo(black_elo)
        if w_elo_val and w_elo_val > 1500:
            add_player(d['White'], counts)
        if b_elo_val and b_elo_val > 1500:
            add_player(d['Black'], counts)
        if i % 10000 == 0:
            print(i)

    with bz2.open("./player_count.csv.bz2", 'wt') as f:
        f.write("player,count\n")
        for p, c in sorted(counts.items(), key = lambda x: x[1], reverse=True):
            f.write(f"{p},{c}\n")

def split_by_player(players):
    games = GamesFile(input_pgn)

    outputs_white = []
    outputs_black = []
    for i, (d, l) in enumerate(games):
        if d['White'] in players:
            outputs_white.append(l)
        elif d['Black'] in players:
            outputs_black.append(l)
        if i % 10000 == 0:
            print("")

# df = pd.read_csv("./player_count.csv.bz2", low_memory=False)

# players_top_df = df.sort_values("count", ascending=False).head(3000)

# print(players_top_df["count"][3000])
# split_by_player(players_top_df["player"])
# 第二步：筛选出对局数不少于200的玩家名单
# qualified_players = [player for player, count in players_count.items() if count >= 200]
# 
# # 第三步：为每个合格玩家创建一个PGN文件（文件名使用玩家姓名，特殊字符替换为下划线）
# player_pgn_paths = {}  # 保存玩家对应的PGN文件路径
# for player in qualified_players:
#     # 清理文件名中不合法字符
#     sanitized_name = re.sub(r'[<>:"/\\\\|?*]', '_', player)  # 将文件名中非法字符替换
#     sanitized_name = sanitized_name.replace(' ', '_')
#     pgn_path = os.path.join(output_dir, f"{sanitized_name}.pgn")
#     player_pgn_paths[player] = pgn_path
#     # 创建空的PGN文件准备写入
#     with open(pgn_path, "w", encoding="utf-8") as f:
#         pass  # 先创建清空文件
# 
# # 第四步：遍历PGN再次，将合格玩家的对局写入各自PGN文件
# with open(input_pgn, encoding="utf-8") as pgn:
#     while True:
#         game = chess.pgn.read_game(pgn)
#         if game is None:
#             break
#         headers = game.headers
#         white_player = headers.get("White")
#         black_player = headers.get("Black")
#         # 如果对局的白方或黑方在合格玩家名单中，则写入对应PGN文件
#         game_text = str(game)  # 将棋局转换为PGN格式文本
#         # 为保证多个对局间有空行分隔，添加两个换行符
#         game_text += "\n\n"
#         if white_player in player_pgn_paths:
#             with open(player_pgn_paths[white_player], "a", encoding="utf-8") as f:
#                 f.write(game_text)
#         if black_player in player_pgn_paths:
#             with open(player_pgn_paths[black_player], "a", encoding="utf-8") as f:
#                 f.write(game_text)