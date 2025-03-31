
from chess_data_parse.pgn_parsering import GamesFile
from chess_data_parse.pgn_to_chunk import parse_elo

games = GamesFile("./player_pgns/A009.pgn.bz2")

for i, (d, _) in enumerate(games):
    white_elo = d['WhiteElo']
    w_elo_val = parse_elo(white_elo)
    print(w_elo_val, d['White'])